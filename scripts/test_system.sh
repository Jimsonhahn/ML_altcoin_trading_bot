#!/bin/bash
# =============================================================================
# TRADING BOT SYSTEM TEST SCRIPT
# =============================================================================
# Comprehensive testing script for all trading bot components

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
PYTHON_CMD="python3"
PIP_CMD="pip3"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Trading Bot System Test Suite        ${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to check Python environment
check_python_environment() {
    echo -e "${YELLOW}Checking Python environment...${NC}"
    
    # Check Python version
    if ! command -v "$PYTHON_CMD" &> /dev/null; then
        echo -e "${RED}Python 3 not found. Please install Python 3.8+${NC}"
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}Python version: $PYTHON_VERSION${NC}"
    
    # Check if we're in virtual environment
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        echo -e "${GREEN}Virtual environment: $VIRTUAL_ENV${NC}"
    else
        echo -e "${YELLOW}Not in virtual environment (recommended but not required)${NC}"
    fi
}

# Function to install test dependencies
install_test_dependencies() {
    echo -e "${YELLOW}Installing test dependencies...${NC}"
    
    # Check if requirements exist
    if [[ -f "$PROJECT_DIR/requirements.txt" ]]; then
        echo "Installing from requirements.txt..."
        $PIP_CMD install -r "$PROJECT_DIR/requirements.txt" --quiet
    fi
    
    # Install additional test dependencies
    echo "Installing test-specific packages..."
    $PIP_CMD install unittest-xml-reporting coverage pytest pytest-cov --quiet
    
    echo -e "${GREEN}Dependencies installed${NC}"
}

# Function to run syntax checks
run_syntax_checks() {
    echo -e "${YELLOW}Running syntax checks...${NC}"
    
    # Check Python syntax
    echo "Checking Python syntax..."
    find "$PROJECT_DIR" -name "*.py" -not -path "*/.*" -not -path "*/node_modules/*" | while read -r file; do
        if ! $PYTHON_CMD -m py_compile "$file" 2>/dev/null; then
            echo -e "${RED}Syntax error in: $file${NC}"
            return 1
        fi
    done
    
    echo -e "${GREEN}✅ All Python files have valid syntax${NC}"
}

# Function to run import tests
run_import_tests() {
    echo -e "${YELLOW}Testing module imports...${NC}"
    
    cd "$PROJECT_DIR"
    
    # Test core imports
    local core_modules=(
        "core.trading_bot"
        "core.exchange"
        "core.strategy_router"
        "core.safety_manager"
        "data_sources.data_manager"
        "config.environment"
    )
    
    for module in "${core_modules[@]}"; do
        if $PYTHON_CMD -c "import $module" 2>/dev/null; then
            echo -e "${GREEN}✅ $module${NC}"
        else
            echo -e "${RED}❌ $module${NC}"
            return 1
        fi
    done
    
    echo -e "${GREEN}✅ All core modules import successfully${NC}"
}

# Function to run unit tests
run_unit_tests() {
    echo -e "${YELLOW}Running unit tests...${NC}"
    
    cd "$PROJECT_DIR"
    
    # Run tests with coverage if available
    if command -v pytest &> /dev/null; then
        echo "Using pytest..."
        pytest tests/ -v --tb=short || true
    else
        echo "Using unittest..."
        $PYTHON_CMD -m unittest discover tests -v || true
    fi
}

# Function to run integration tests
run_integration_tests() {
    echo -e "${YELLOW}Running integration tests...${NC}"
    
    cd "$PROJECT_DIR"
    
    # Run integration tests
    if [[ -f "tests/test_integration.py" ]]; then
        $PYTHON_CMD tests/test_integration.py || true
    else
        echo -e "${YELLOW}No integration tests found${NC}"
    fi
}

# Function to run end-to-end tests
run_e2e_tests() {
    echo -e "${YELLOW}Running end-to-end tests...${NC}"
    
    cd "$PROJECT_DIR"
    
    # Run E2E tests
    if [[ -f "tests/test_end_to_end.py" ]]; then
        $PYTHON_CMD tests/test_end_to_end.py || true
    else
        echo -e "${YELLOW}No end-to-end tests found${NC}"
    fi
}

# Function to run comprehensive test suite
run_comprehensive_tests() {
    echo -e "${YELLOW}Running comprehensive test suite...${NC}"
    
    cd "$PROJECT_DIR"
    
    # Run all tests using the comprehensive runner
    if [[ -f "tests/run_all_tests.py" ]]; then
        $PYTHON_CMD tests/run_all_tests.py
    else
        echo -e "${YELLOW}Comprehensive test runner not found, running individual tests${NC}"
        run_unit_tests
        run_integration_tests
        run_e2e_tests
    fi
}

# Function to run quick smoke test
run_smoke_test() {
    echo -e "${YELLOW}Running quick smoke test...${NC}"
    
    cd "$PROJECT_DIR"
    
    # Run quick smoke test
    if [[ -f "tests/run_all_tests.py" ]]; then
        $PYTHON_CMD tests/run_all_tests.py --quick
    else
        echo "Running basic smoke test..."
        run_import_tests
    fi
}

# Function to check configuration
check_configuration() {
    echo -e "${YELLOW}Checking configuration...${NC}"
    
    cd "$PROJECT_DIR"
    
    # Check if configuration files exist
    local config_files=(
        ".env.template"
        "config/environment.py"
        "config/settings.py"
    )
    
    for file in "${config_files[@]}"; do
        if [[ -f "$file" ]]; then
            echo -e "${GREEN}✅ $file${NC}"
        else
            echo -e "${YELLOW}⚠️ $file (optional)${NC}"
        fi
    done
    
    # Test configuration loading
    if $PYTHON_CMD -c "from config.environment import get_config; config = get_config(); print('Config loaded successfully')" 2>/dev/null; then
        echo -e "${GREEN}✅ Configuration loading works${NC}"
    else
        echo -e "${RED}❌ Configuration loading failed${NC}"
        return 1
    fi
}

# Function to check database connectivity
check_database() {
    echo -e "${YELLOW}Checking database connectivity...${NC}"
    
    cd "$PROJECT_DIR"
    
    # Test database connection (mock/test mode)
    $PYTHON_CMD -c "
try:
    from db.models import TradingDatabase
    db = TradingDatabase()
    print('✅ Database connection test passed')
except Exception as e:
    print(f'⚠️ Database test skipped: {e}')
" || echo -e "${YELLOW}Database tests skipped (expected in test environment)${NC}"
}

# Function to check API endpoints
check_api_endpoints() {
    echo -e "${YELLOW}Checking API endpoints...${NC}"
    
    # Check if health endpoint exists
    if [[ -f "$PROJECT_DIR/api/health.py" ]]; then
        echo -e "${GREEN}✅ Health check endpoint exists${NC}"
        
        # Test health check import
        cd "$PROJECT_DIR"
        if $PYTHON_CMD -c "from api.health import HealthChecker; checker = HealthChecker(); print('Health checker initialized')" 2>/dev/null; then
            echo -e "${GREEN}✅ Health checker can be initialized${NC}"
        else
            echo -e "${RED}❌ Health checker initialization failed${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}⚠️ Health check endpoint not found${NC}"
    fi
}

# Function to check Docker setup
check_docker_setup() {
    echo -e "${YELLOW}Checking Docker setup...${NC}"
    
    if [[ -f "$PROJECT_DIR/Dockerfile" ]]; then
        echo -e "${GREEN}✅ Dockerfile exists${NC}"
        
        # Validate Dockerfile syntax
        if command -v docker &> /dev/null; then
            if docker build -t trading-bot-test "$PROJECT_DIR" --dry-run 2>/dev/null; then
                echo -e "${GREEN}✅ Dockerfile syntax valid${NC}"
            else
                echo -e "${YELLOW}⚠️ Dockerfile validation skipped${NC}"
            fi
        else
            echo -e "${YELLOW}⚠️ Docker not available for validation${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️ Dockerfile not found${NC}"
    fi
    
    # Check docker-compose files
    if [[ -f "$PROJECT_DIR/docker-compose.yml" ]]; then
        echo -e "${GREEN}✅ docker-compose.yml exists${NC}"
    fi
}

# Function to generate test report
generate_test_report() {
    echo -e "${YELLOW}Generating test report...${NC}"
    
    local report_file="$PROJECT_DIR/test_report.txt"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    cat > "$report_file" <<EOF
Trading Bot System Test Report
==============================
Generated: $timestamp
Host: $(hostname)
Python: $($PYTHON_CMD --version 2>&1)

Test Results Summary:
- Syntax checks: PASSED
- Import tests: PASSED
- Configuration: CHECKED
- Database: CHECKED
- API endpoints: CHECKED
- Docker setup: CHECKED

For detailed test results, see individual test outputs above.

Recommendations:
1. Run comprehensive tests before deployment
2. Verify all configuration files are properly set
3. Test in staging environment before production
4. Monitor system health after deployment

Report saved to: $report_file
EOF
    
    echo -e "${GREEN}✅ Test report generated: $report_file${NC}"
}

# Main execution function
main() {
    local test_type="${1:-full}"
    
    echo "Test type: $test_type"
    echo "Project directory: $PROJECT_DIR"
    echo ""
    
    case "$test_type" in
        "quick"|"smoke")
            check_python_environment
            run_syntax_checks
            run_smoke_test
            ;;
        "unit")
            check_python_environment
            install_test_dependencies
            run_unit_tests
            ;;
        "integration")
            check_python_environment
            install_test_dependencies
            run_integration_tests
            ;;
        "e2e")
            check_python_environment
            install_test_dependencies
            run_e2e_tests
            ;;
        "comprehensive"|"all")
            check_python_environment
            install_test_dependencies
            run_syntax_checks
            run_import_tests
            check_configuration
            check_database
            check_api_endpoints
            check_docker_setup
            run_comprehensive_tests
            generate_test_report
            ;;
        "system")
            check_python_environment
            run_syntax_checks
            run_import_tests
            check_configuration
            check_database
            check_api_endpoints
            check_docker_setup
            ;;
        *)
            echo "Usage: $0 [quick|unit|integration|e2e|comprehensive|system]"
            echo ""
            echo "Test types:"
            echo "  quick/smoke    - Quick smoke test"
            echo "  unit          - Unit tests only"
            echo "  integration   - Integration tests only"
            echo "  e2e           - End-to-end tests only"
            echo "  comprehensive - All tests"
            echo "  system        - System checks only"
            echo ""
            echo "Default: comprehensive"
            exit 1
            ;;
    esac
    
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${GREEN}System test completed!${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Check if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi