#!/bin/bash
# =============================================================================
# SYSTEMD SERVICE INSTALLATION SCRIPT
# =============================================================================
# Install and configure trading bot as systemd service for 24/7 operation

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_NAME="trading-bot"
SERVICE_USER="trading-bot"
INSTALL_DIR="/opt/trading-bot"
LOG_DIR="/var/log/trading-bot"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Trading Bot Systemd Service Setup    ${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}This script must be run as root (use sudo)${NC}"
   exit 1
fi

# Function to create user
create_service_user() {
    echo -e "${YELLOW}Creating service user: $SERVICE_USER${NC}"
    
    if id "$SERVICE_USER" &>/dev/null; then
        echo -e "${GREEN}User $SERVICE_USER already exists${NC}"
    else
        useradd --system --shell /bin/false --home-dir "$INSTALL_DIR" --create-home "$SERVICE_USER"
        echo -e "${GREEN}User $SERVICE_USER created${NC}"
    fi
}

# Function to create directories
create_directories() {
    echo -e "${YELLOW}Creating directories...${NC}"
    
    # Create installation directory
    mkdir -p "$INSTALL_DIR"
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    mkdir -p "$LOG_DIR/archive"
    
    # Create data directories
    mkdir -p "$INSTALL_DIR/data"
    mkdir -p "$INSTALL_DIR/logs"
    mkdir -p "$INSTALL_DIR/backups"
    
    echo -e "${GREEN}Directories created${NC}"
}

# Function to copy files
copy_project_files() {
    echo -e "${YELLOW}Copying project files to $INSTALL_DIR...${NC}"
    
    # Copy all project files except unnecessary ones
    rsync -av \
        --exclude='.git/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='.pytest_cache/' \
        --exclude='node_modules/' \
        --exclude='.env.dev' \
        --exclude='data/cache/' \
        --exclude='logs/' \
        "$PROJECT_DIR/" "$INSTALL_DIR/"
    
    echo -e "${GREEN}Project files copied${NC}"
}

# Function to set permissions
set_permissions() {
    echo -e "${YELLOW}Setting permissions...${NC}"
    
    # Set ownership
    chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
    chown -R "$SERVICE_USER:$SERVICE_USER" "$LOG_DIR"
    
    # Set file permissions
    find "$INSTALL_DIR" -type f -name "*.py" -exec chmod 755 {} \;
    find "$INSTALL_DIR" -type f -name "*.sh" -exec chmod 755 {} \;
    find "$INSTALL_DIR" -type f -name "*.json" -exec chmod 644 {} \;
    find "$INSTALL_DIR" -type f -name ".env*" -exec chmod 600 {} \;
    
    # Set directory permissions
    find "$INSTALL_DIR" -type d -exec chmod 755 {} \;
    find "$LOG_DIR" -type d -exec chmod 755 {} \;
    
    echo -e "${GREEN}Permissions set${NC}"
}

# Function to install Python dependencies
install_dependencies() {
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    
    # Install system Python packages
    apt-get update
    apt-get install -y python3 python3-pip python3-venv
    
    # Create virtual environment
    sudo -u "$SERVICE_USER" python3 -m venv "$INSTALL_DIR/venv"
    
    # Install Python packages
    sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install --upgrade pip
    
    if [[ -f "$INSTALL_DIR/requirements.txt" ]]; then
        sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt"
    else
        # Install common packages
        sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install \
            ccxt pandas numpy scikit-learn xgboost \
            flask flask-cors redis psycopg2-binary \
            python-telegram-bot schedule requests \
            websocket-client asyncio aiohttp \
            python-dotenv pyyaml configargparse \
            prometheus-client psutil
    fi
    
    echo -e "${GREEN}Dependencies installed${NC}"
}

# Function to install systemd service
install_systemd_service() {
    echo -e "${YELLOW}Installing systemd service...${NC}"
    
    # Update service file paths
    sed -i "s|/usr/local/bin/python3|$INSTALL_DIR/venv/bin/python|g" "$SCRIPT_DIR/trading-bot.service"
    sed -i "s|/opt/trading-bot|$INSTALL_DIR|g" "$SCRIPT_DIR/trading-bot.service"
    
    # Copy service file
    cp "$SCRIPT_DIR/trading-bot.service" "/etc/systemd/system/"
    
    # Reload systemd
    systemctl daemon-reload
    
    # Enable service
    systemctl enable "$SERVICE_NAME"
    
    echo -e "${GREEN}Systemd service installed and enabled${NC}"
}

# Function to configure logrotate
configure_logrotate() {
    echo -e "${YELLOW}Configuring log rotation...${NC}"
    
    cat > "/etc/logrotate.d/trading-bot" <<EOF
$LOG_DIR/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 $SERVICE_USER $SERVICE_USER
    postrotate
        systemctl reload $SERVICE_NAME || true
    endscript
}
EOF
    
    echo -e "${GREEN}Log rotation configured${NC}"
}

# Function to create monitoring scripts
create_monitoring_scripts() {
    echo -e "${YELLOW}Creating monitoring scripts...${NC}"
    
    cat > "$INSTALL_DIR/scripts/system_monitor.sh" <<'EOF'
#!/bin/bash
# System monitoring script for trading bot

SERVICE_NAME="trading-bot"
LOG_FILE="/var/log/trading-bot/monitor.log"

echo "$(date): Checking trading bot status..." >> "$LOG_FILE"

if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo "$(date): Trading bot is running" >> "$LOG_FILE"
    
    # Check memory usage
    MEMORY=$(systemctl show "$SERVICE_NAME" --property=MemoryCurrent --value)
    echo "$(date): Memory usage: $MEMORY bytes" >> "$LOG_FILE"
    
    # Check if API is responding
    if curl -f -s http://localhost:8080/health >/dev/null; then
        echo "$(date): API health check passed" >> "$LOG_FILE"
    else
        echo "$(date): WARNING: API health check failed" >> "$LOG_FILE"
        # Could send alert here
    fi
else
    echo "$(date): ERROR: Trading bot is not running!" >> "$LOG_FILE"
    # Restart service
    systemctl start "$SERVICE_NAME"
    echo "$(date): Attempted to restart trading bot" >> "$LOG_FILE"
fi
EOF

    chmod +x "$INSTALL_DIR/scripts/system_monitor.sh"
    chown "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR/scripts/system_monitor.sh"
    
    echo -e "${GREEN}Monitoring scripts created${NC}"
}

# Function to setup cron jobs
setup_cron() {
    echo -e "${YELLOW}Setting up cron jobs...${NC}"
    
    # Add monitoring cron job
    (crontab -u "$SERVICE_USER" -l 2>/dev/null; echo "*/5 * * * * $INSTALL_DIR/scripts/system_monitor.sh") | crontab -u "$SERVICE_USER" -
    
    # Add daily backup cron job
    (crontab -u "$SERVICE_USER" -l 2>/dev/null; echo "0 2 * * * $INSTALL_DIR/scripts/backup.sh") | crontab -u "$SERVICE_USER" -
    
    echo -e "${GREEN}Cron jobs configured${NC}"
}

# Function to create firewall rules
configure_firewall() {
    echo -e "${YELLOW}Configuring firewall...${NC}"
    
    if command -v ufw >/dev/null 2>&1; then
        # Allow API port
        ufw allow 8080/tcp comment "Trading Bot API"
        
        # Allow health check port
        ufw allow 9090/tcp comment "Trading Bot Metrics"
        
        echo -e "${GREEN}Firewall rules added${NC}"
    else
        echo -e "${YELLOW}UFW not found, skipping firewall configuration${NC}"
    fi
}

# Function to validate installation
validate_installation() {
    echo -e "${YELLOW}Validating installation...${NC}"
    
    # Check if service is loaded
    if systemctl list-unit-files | grep -q "$SERVICE_NAME"; then
        echo -e "${GREEN}✓ Service file installed${NC}"
    else
        echo -e "${RED}✗ Service file not found${NC}"
        return 1
    fi
    
    # Check if user exists
    if id "$SERVICE_USER" &>/dev/null; then
        echo -e "${GREEN}✓ Service user exists${NC}"
    else
        echo -e "${RED}✗ Service user not found${NC}"
        return 1
    fi
    
    # Check if directories exist
    if [[ -d "$INSTALL_DIR" ]]; then
        echo -e "${GREEN}✓ Installation directory exists${NC}"
    else
        echo -e "${RED}✗ Installation directory not found${NC}"
        return 1
    fi
    
    # Check permissions
    if [[ -O "$INSTALL_DIR" ]] || [[ $(stat -c '%U' "$INSTALL_DIR") == "$SERVICE_USER" ]]; then
        echo -e "${GREEN}✓ Permissions are correct${NC}"
    else
        echo -e "${RED}✗ Permission issues detected${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Installation validation completed${NC}"
}

# Main installation process
main() {
    echo -e "${BLUE}Starting installation...${NC}"
    
    create_service_user
    create_directories
    copy_project_files
    install_dependencies
    set_permissions
    install_systemd_service
    configure_logrotate
    create_monitoring_scripts
    setup_cron
    configure_firewall
    validate_installation
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}Installation completed successfully!${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Edit $INSTALL_DIR/.env.production with your API keys"
    echo "2. Start the service: sudo systemctl start $SERVICE_NAME"
    echo "3. Check status: sudo systemctl status $SERVICE_NAME"
    echo "4. View logs: sudo journalctl -u $SERVICE_NAME -f"
    echo "5. Check health: curl http://localhost:8080/health"
    echo
    echo -e "${YELLOW}Service commands:${NC}"
    echo "Start:   sudo systemctl start $SERVICE_NAME"
    echo "Stop:    sudo systemctl stop $SERVICE_NAME"
    echo "Restart: sudo systemctl restart $SERVICE_NAME"
    echo "Status:  sudo systemctl status $SERVICE_NAME"
    echo "Logs:    sudo journalctl -u $SERVICE_NAME -f"
}

# Run main function
main "$@"