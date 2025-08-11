#!/bin/bash
# 🔄 Update Script (run on server)

echo "🔄 Updating Janics Freedom Factory..."

# Pull latest changes
git pull origin main

# Activate virtual environment
source venv/bin/activate

# Update dependencies
pip install -r requirements.txt

# Restart service
sudo systemctl restart janics_bot

# Show status
sudo systemctl status janics_bot --no-pager

echo "✅ Update complete!"
