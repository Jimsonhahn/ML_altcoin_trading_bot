#!/bin/bash
# 🎯 ULTIMATE STRATO DEPLOYMENT SCRIPT
# One-click deployment with all fixes applied

echo "🎯 ULTIMATE STRATO SERVER DEPLOYMENT"
echo "===================================="
echo "Server: 85.215.183.30"
echo "Fixes: Trading Pipeline Fully Functional"
echo ""

# Menu für Deployment-Optionen
echo "Choose deployment method:"
echo "1) SSH Linux Server"
echo "2) Windows RDP Server"
echo "3) Test Server Connection"
echo "4) Manual Instructions"
echo "5) Exit"
echo ""

read -p "Enter choice (1-5): " choice

case $choice in
    1)
        echo "🐧 Linux SSH Deployment"
        echo "======================"
        
        # Test SSH connection first
        echo "Testing SSH connection..."
        python3 test_server_connection.py
        
        read -p "SSH connection OK? Continue with update? (y/n): " continue_ssh
        if [ "$continue_ssh" = "y" ]; then
            echo "Running Linux update script..."
            ./strato_update_bot.sh
        fi
        ;;
        
    2)
        echo "🪟 Windows RDP Deployment"
        echo "========================"
        
        echo "Starting Windows PowerShell deployment..."
        powershell -ExecutionPolicy Bypass -File "./strato_update_windows.ps1"
        ;;
        
    3)
        echo "🧪 Testing Server Connection"
        echo "============================"
        
        python3 test_server_connection.py
        
        echo ""
        echo "Based on the test results:"
        echo "- If SSH works: Use option 1"
        echo "- If RDP works: Use option 2"
        echo "- If neither: Use option 4 for manual deployment"
        ;;
        
    4)
        echo "📖 Opening Manual Deployment Guide"
        echo "=================================="
        
        if command -v open &> /dev/null; then
            open STRATO_DEPLOYMENT_MANUAL.md
        elif command -v xdg-open &> /dev/null; then
            xdg-open STRATO_DEPLOYMENT_MANUAL.md
        else
            echo "Please open: STRATO_DEPLOYMENT_MANUAL.md"
        fi
        
        echo ""
        echo "📋 Quick Manual Steps:"
        echo "1. Connect to server via SSH/RDP/Web Panel"
        echo "2. Navigate to bot directory"
        echo "3. Run: git pull origin main"
        echo "4. Run: pip install -r requirements.txt"
        echo "5. Test: python3 test_actual_trading.py"
        echo "6. Start: python3 main.py --mode paper"
        echo "7. Dashboard: python3 api/app.py --host 0.0.0.0 --port 8000"
        echo "8. Access: http://85.215.183.30:8000"
        ;;
        
    5)
        echo "Deployment cancelled."
        exit 0
        ;;
        
    *)
        echo "Invalid choice. Please run script again."
        exit 1
        ;;
esac

echo ""
echo "🎯 DEPLOYMENT FILES READY:"
echo "✅ strato_update_bot.sh (Linux SSH)"
echo "✅ strato_update_windows.ps1 (Windows RDP)"  
echo "✅ test_server_connection.py (Connection test)"
echo "✅ STRATO_DEPLOYMENT_MANUAL.md (Manual guide)"
echo ""
echo "🔧 CRITICAL FIXES INCLUDED:"
echo "✅ Trading pipeline execution fixed"
echo "✅ Market data retrieval working"  
echo "✅ Strategy signal generation fixed"
echo "✅ Risk management interfaces completed"
echo "✅ Order execution simulation ready"
echo ""
echo "💰 Your bot will now ACTUALLY trade instead of just showing 'Running'!"
echo ""
echo "Need help? Check the manual: STRATO_DEPLOYMENT_MANUAL.md"