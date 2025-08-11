#!/bin/bash
# =============================================================================
# TRADING BOT BACKUP SCRIPT
# =============================================================================
# Automated backup script for trading bot data and configuration

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="${BACKUP_DIR:-/opt/trading-bot/backups}"
INSTALL_DIR="${INSTALL_DIR:-/opt/trading-bot}"
DB_NAME="${DB_NAME:-trading_bot_production}"
DB_USER="${DB_USER:-trader_prod}"

# Backup retention settings
DAILY_RETENTION=7
WEEKLY_RETENTION=4
MONTHLY_RETENTION=6

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging
LOG_FILE="/var/log/trading-bot/backup.log"
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Trading Bot Backup - $(date)${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to create timestamped backup directory
create_backup_dir() {
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local backup_type="$1"
    
    CURRENT_BACKUP_DIR="$BACKUP_DIR/${backup_type}_$timestamp"
    mkdir -p "$CURRENT_BACKUP_DIR"
    
    echo -e "${GREEN}Created backup directory: $CURRENT_BACKUP_DIR${NC}"
}

# Function to backup configuration files
backup_config() {
    echo -e "${YELLOW}Backing up configuration files...${NC}"
    
    local config_dir="$CURRENT_BACKUP_DIR/config"
    mkdir -p "$config_dir"
    
    # Backup environment files (excluding sensitive data)
    if [[ -f "$INSTALL_DIR/.env.production" ]]; then
        # Create sanitized version without sensitive keys
        grep -v -E "(API_KEY|SECRET|PASSWORD|TOKEN)" "$INSTALL_DIR/.env.production" > "$config_dir/env.production.sanitized"
        
        # Backup actual file (will be encrypted)
        cp "$INSTALL_DIR/.env.production" "$config_dir/.env.production"
    fi
    
    # Backup other config files
    if [[ -d "$INSTALL_DIR/config" ]]; then
        cp -r "$INSTALL_DIR/config" "$config_dir/"
    fi
    
    # Backup PM2 and systemd configs
    if [[ -f "$INSTALL_DIR/scripts/pm2-config.json" ]]; then
        cp "$INSTALL_DIR/scripts/pm2-config.json" "$config_dir/"
    fi
    
    if [[ -f "/etc/systemd/system/trading-bot.service" ]]; then
        cp "/etc/systemd/system/trading-bot.service" "$config_dir/"
    fi
    
    echo -e "${GREEN}Configuration files backed up${NC}"
}

# Function to backup trading data
backup_trading_data() {
    echo -e "${YELLOW}Backing up trading data...${NC}"
    
    local data_dir="$CURRENT_BACKUP_DIR/data"
    mkdir -p "$data_dir"
    
    # Backup trading data directory
    if [[ -d "$INSTALL_DIR/data" ]]; then
        cp -r "$INSTALL_DIR/data" "$data_dir/"
    fi
    
    # Backup logs (last 7 days)
    if [[ -d "/var/log/trading-bot" ]]; then
        find "/var/log/trading-bot" -name "*.log" -mtime -7 -exec cp {} "$data_dir/" \;
    fi
    
    echo -e "${GREEN}Trading data backed up${NC}"
}

# Function to backup database
backup_database() {
    echo -e "${YELLOW}Backing up database...${NC}"
    
    local db_dir="$CURRENT_BACKUP_DIR/database"
    mkdir -p "$db_dir"
    
    # Check if database connection is available
    if command -v pg_dump >/dev/null 2>&1; then
        # PostgreSQL backup
        local timestamp=$(date +"%Y%m%d_%H%M%S")
        local dump_file="$db_dir/trading_bot_${timestamp}.sql"
        
        if pg_dump -h "${DB_HOST:-localhost}" -U "$DB_USER" -d "$DB_NAME" > "$dump_file" 2>/dev/null; then
            echo -e "${GREEN}Database dumped to: $dump_file${NC}"
            
            # Compress the dump
            gzip "$dump_file"
            echo -e "${GREEN}Database backup compressed${NC}"
        else
            echo -e "${YELLOW}Database backup failed or skipped (no connection)${NC}"
        fi
    else
        echo -e "${YELLOW}pg_dump not available, skipping database backup${NC}"
    fi
}

# Function to backup source code
backup_source() {
    echo -e "${YELLOW}Backing up source code...${NC}"
    
    local source_dir="$CURRENT_BACKUP_DIR/source"
    mkdir -p "$source_dir"
    
    # Backup source code (excluding cache and logs)
    rsync -av \
        --exclude='.git/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='.pytest_cache/' \
        --exclude='node_modules/' \
        --exclude='data/cache/' \
        --exclude='logs/' \
        --exclude='backups/' \
        "$INSTALL_DIR/" "$source_dir/"
    
    echo -e "${GREEN}Source code backed up${NC}"
}

# Function to create backup manifest
create_manifest() {
    echo -e "${YELLOW}Creating backup manifest...${NC}"
    
    local manifest_file="$CURRENT_BACKUP_DIR/backup_manifest.json"
    
    cat > "$manifest_file" <<EOF
{
    "backup_timestamp": "$(date -Iseconds)",
    "backup_type": "$BACKUP_TYPE",
    "hostname": "$(hostname)",
    "system_info": {
        "os": "$(uname -s)",
        "kernel": "$(uname -r)",
        "architecture": "$(uname -m)"
    },
    "trading_bot_info": {
        "version": "1.0.0",
        "environment": "$(grep NODE_ENV $INSTALL_DIR/.env.production | cut -d'=' -f2 || echo 'unknown')",
        "install_dir": "$INSTALL_DIR",
        "backup_size_mb": $(du -sm "$CURRENT_BACKUP_DIR" | cut -f1)
    },
    "backup_contents": {
        "config_files": $(find "$CURRENT_BACKUP_DIR/config" -type f 2>/dev/null | wc -l || echo 0),
        "data_files": $(find "$CURRENT_BACKUP_DIR/data" -type f 2>/dev/null | wc -l || echo 0),
        "source_files": $(find "$CURRENT_BACKUP_DIR/source" -type f 2>/dev/null | wc -l || echo 0),
        "database_dumps": $(find "$CURRENT_BACKUP_DIR/database" -name "*.sql.gz" 2>/dev/null | wc -l || echo 0)
    },
    "verification": {
        "md5_checksums": "$(find "$CURRENT_BACKUP_DIR" -type f -exec md5sum {} \; | md5sum | cut -d' ' -f1)"
    }
}
EOF
    
    echo -e "${GREEN}Backup manifest created${NC}"
}

# Function to encrypt sensitive files
encrypt_sensitive_files() {
    echo -e "${YELLOW}Encrypting sensitive files...${NC}"
    
    # Check if GPG is available
    if command -v gpg >/dev/null 2>&1; then
        # Get recipient from environment or use default
        local recipient="${BACKUP_GPG_RECIPIENT:-trading-bot-backup@localhost}"
        
        # Encrypt environment files
        if [[ -f "$CURRENT_BACKUP_DIR/config/.env.production" ]]; then
            if gpg --trust-model always --encrypt -r "$recipient" "$CURRENT_BACKUP_DIR/config/.env.production" 2>/dev/null; then
                rm "$CURRENT_BACKUP_DIR/config/.env.production"
                echo -e "${GREEN}Environment file encrypted${NC}"
            else
                echo -e "${YELLOW}Failed to encrypt environment file${NC}"
            fi
        fi
    else
        echo -e "${YELLOW}GPG not available, skipping encryption${NC}"
    fi
}

# Function to compress backup
compress_backup() {
    echo -e "${YELLOW}Compressing backup...${NC}"
    
    local backup_archive="$BACKUP_DIR/$(basename "$CURRENT_BACKUP_DIR").tar.gz"
    
    # Create compressed archive
    tar -czf "$backup_archive" -C "$BACKUP_DIR" "$(basename "$CURRENT_BACKUP_DIR")"
    
    if [[ -f "$backup_archive" ]]; then
        # Remove uncompressed directory
        rm -rf "$CURRENT_BACKUP_DIR"
        
        # Update current backup dir reference
        CURRENT_BACKUP_DIR="$backup_archive"
        
        local size_mb=$(du -m "$backup_archive" | cut -f1)
        echo -e "${GREEN}Backup compressed: $backup_archive (${size_mb}MB)${NC}"
    else
        echo -e "${RED}Failed to create compressed backup${NC}"
        return 1
    fi
}

# Function to cleanup old backups
cleanup_old_backups() {
    echo -e "${YELLOW}Cleaning up old backups...${NC}"
    
    # Remove old daily backups
    find "$BACKUP_DIR" -name "daily_*.tar.gz" -mtime +$DAILY_RETENTION -delete 2>/dev/null || true
    
    # Remove old weekly backups
    find "$BACKUP_DIR" -name "weekly_*.tar.gz" -mtime +$((WEEKLY_RETENTION * 7)) -delete 2>/dev/null || true
    
    # Remove old monthly backups
    find "$BACKUP_DIR" -name "monthly_*.tar.gz" -mtime +$((MONTHLY_RETENTION * 30)) -delete 2>/dev/null || true
    
    echo -e "${GREEN}Old backups cleaned up${NC}"
}

# Function to verify backup
verify_backup() {
    echo -e "${YELLOW}Verifying backup...${NC}"
    
    if [[ -f "$CURRENT_BACKUP_DIR" ]]; then
        # Test archive integrity
        if tar -tzf "$CURRENT_BACKUP_DIR" >/dev/null 2>&1; then
            echo -e "${GREEN}Backup archive is valid${NC}"
            return 0
        else
            echo -e "${RED}Backup archive is corrupted${NC}"
            return 1
        fi
    else
        echo -e "${RED}Backup file not found${NC}"
        return 1
    fi
}

# Function to send backup notification
send_notification() {
    local status="$1"
    local message="$2"
    
    # Send notification if Telegram is configured
    if [[ -f "$INSTALL_DIR/.env.production" ]]; then
        source "$INSTALL_DIR/.env.production"
        
        if [[ -n "${TELEGRAM_BOT_TOKEN:-}" && -n "${TELEGRAM_CHAT_ID:-}" ]]; then
            local emoji="✅"
            [[ "$status" != "success" ]] && emoji="❌"
            
            local notification_text="$emoji Trading Bot Backup - $status

$message

Time: $(date)
Host: $(hostname)
Backup Type: $BACKUP_TYPE"

            curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
                -d chat_id="${TELEGRAM_CHAT_ID}" \
                -d text="$notification_text" >/dev/null 2>&1 || true
        fi
    fi
}

# Main backup function
perform_backup() {
    local backup_type="$1"
    
    BACKUP_TYPE="$backup_type"
    
    echo -e "${BLUE}Starting $backup_type backup...${NC}"
    
    # Create backup directory
    create_backup_dir "$backup_type"
    
    # Perform backup steps
    backup_config
    backup_trading_data
    backup_database
    backup_source
    create_manifest
    encrypt_sensitive_files
    compress_backup
    
    # Verify backup
    if verify_backup; then
        echo -e "${GREEN}Backup completed successfully${NC}"
        
        local size_mb=$(du -m "$CURRENT_BACKUP_DIR" | cut -f1)
        send_notification "success" "Backup completed successfully (${size_mb}MB)"
        
        return 0
    else
        echo -e "${RED}Backup verification failed${NC}"
        send_notification "failed" "Backup verification failed"
        return 1
    fi
}

# Function to determine backup type
determine_backup_type() {
    local day_of_week=$(date +%u)  # 1=Monday, 7=Sunday
    local day_of_month=$(date +%d)
    
    # Monthly backup on 1st of month
    if [[ "$day_of_month" == "01" ]]; then
        echo "monthly"
    # Weekly backup on Sundays
    elif [[ "$day_of_week" == "7" ]]; then
        echo "weekly"
    # Daily backup otherwise
    else
        echo "daily"
    fi
}

# Main execution
main() {
    # Ensure backup directory exists
    mkdir -p "$BACKUP_DIR"
    
    # Determine backup type if not specified
    local backup_type="${1:-$(determine_backup_type)}"
    
    # Perform backup
    if perform_backup "$backup_type"; then
        # Cleanup old backups
        cleanup_old_backups
        
        echo -e "${BLUE}========================================${NC}"
        echo -e "${GREEN}Backup process completed successfully${NC}"
        echo -e "${BLUE}========================================${NC}"
        
        exit 0
    else
        echo -e "${BLUE}========================================${NC}"
        echo -e "${RED}Backup process failed${NC}"
        echo -e "${BLUE}========================================${NC}"
        
        exit 1
    fi
}

# Check if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi