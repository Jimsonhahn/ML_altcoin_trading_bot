# 🎯 STRATO SERVER MANUAL DEPLOYMENT GUIDE

## ✅ Server Status: REACHABLE (Ping successful)
Server IP: `85.215.183.30`

## 🔍 Current Issues Detected:
- ❌ SSH connection timeout (Port 22)
- ❌ HTTP services not responding (Ports 8000, 8002)

---

## 📋 DEPLOYMENT OPTIONS

### Option 1: Linux Server (SSH Access)

#### Step 1: Connect via SSH
```bash
# Try different SSH ports if 22 is blocked
ssh trading@85.215.183.30
# Or try with custom port:
ssh -p 2222 trading@85.215.183.30
```

#### Step 2: Update Bot (if SSH works)
```bash
# Run our automated update script
./strato_update_bot.sh
```

#### Step 3: Manual Update (if automated script fails)
```bash
# SSH to server and run:
cd /home/trading/altcoin_trading_bot  # Or your bot directory

# Pull latest fixes
git pull origin main

# Update dependencies  
source venv/bin/activate
pip install -r requirements.txt

# Test the fixes
python3 test_actual_trading.py

# Start bot with fixes
python3 main.py --mode paper --strategy momentum &

# Start dashboard
python3 api/app.py --host 0.0.0.0 --port 8000 &
```

---

### Option 2: Windows Server (RDP Access)

#### Step 1: Connect via Remote Desktop
```
Server: 85.215.183.30:3389
Username: administrator (or your username)
Password: [Your RDP password]
```

#### Step 2: Use PowerShell Script
```powershell
# Run our Windows update script
.\strato_update_windows.ps1
```

#### Step 3: Manual Update (if script fails)
1. **Open Command Prompt as Administrator**
2. **Navigate to bot directory:**
   ```cmd
   cd C:\TradingBot
   ```

3. **Stop existing processes:**
   ```cmd
   taskkill /f /im python.exe
   ```

4. **Update from Git:**
   ```cmd
   git pull origin main
   ```

5. **Update dependencies:**
   ```cmd
   call venv\Scripts\activate.bat
   pip install -r requirements.txt
   ```

6. **Test the fixes:**
   ```cmd
   python test_actual_trading.py
   ```

7. **Start bot with fixes:**
   ```cmd
   start python main.py --mode paper --strategy momentum
   start python api/app.py --host 0.0.0.0 --port 8000
   ```

---

### Option 3: Web Panel / Control Panel Access

If Strato provides a web-based control panel:

1. **Login to Strato Control Panel**
2. **Navigate to File Manager**
3. **Upload these files to your bot directory:**
   - `debug_trading_execution.py`
   - `simple_trading_debug.py` 
   - `test_actual_trading.py`
   - All updated Python files

4. **Use Terminal/Console (if available) to run:**
   ```bash
   cd /path/to/bot
   python3 test_actual_trading.py
   ```

---

## 🎯 CRITICAL FIXES TO VERIFY

After deployment, ensure these fixes are active:

### ✅ **Fix 1: Import Paths**
Check that files contain `from analysis.performance_tracker` (not `Analysis.`)

### ✅ **Fix 2: Async Methods** 
Verify `strategies/momentum.py` has `async def calculate_signal()`

### ✅ **Fix 3: Data Manager**
Check `data_sources/data_manager.py` has `async def fetch_data()`

### ✅ **Fix 4: Risk Manager**
Verify `core/risk_manager.py` has `can_open_position()` method

### ✅ **Fix 5: Order Manager** 
Check `core/order_manager.py` has `simulate_order()` method

---

## 🧪 TESTING THE DEPLOYMENT

### Test 1: Run Pipeline Test
```bash
python3 test_actual_trading.py
```

**Expected Output:**
```
✅ Bot created successfully
✅ Strategy signal generated
✅ Risk check passed  
✅ Position calculated
✅ Order simulated
```

### Test 2: Dashboard Access
1. **Start Dashboard:**
   ```bash
   python3 api/app.py --host 0.0.0.0 --port 8000
   ```

2. **Access in Browser:**
   ```
   http://85.215.183.30:8000
   ```

3. **Verify Dashboard Shows:**
   - Bot status: "Running"
   - Trading activity (not just "Running" without trades)
   - Recent signals and trades

### Test 3: Pipeline Debug
```bash
python3 simple_trading_debug.py
```

**Should show:**
```
✅ Paper trading engine available
✅ Exchange module available  
✅ TradingBot available
✅ Bot instance created successfully
✅ Strategies loaded
```

---

## 🚨 TROUBLESHOOTING

### Issue: "Analysis module not found"
**Fix:** Verify import paths were updated from `Analysis.` to `analysis.`

### Issue: "cannot unpack non-iterable coroutine"  
**Fix:** Verify strategy methods are `async def` and properly awaited

### Issue: "RiskManager object has no attribute 'can_open_position'"
**Fix:** Verify `core/risk_manager.py` was updated with new methods

### Issue: Bot shows "Running" but no trades
**Fix:** Run `python3 test_actual_trading.py` to verify pipeline

---

## 📱 DASHBOARD CONFIGURATION

After successful deployment:

1. **Edit Dashboard Config** (if needed):
   ```javascript
   // In dashboard files, set:
   const API_BASE = 'http://85.215.183.30:8000';
   ```

2. **Mobile Access:**
   ```
   http://85.215.183.30:8000/mobile
   ```

3. **API Health Check:**
   ```
   http://85.215.183.30:8000/health
   ```

---

## 💰 SUCCESS INDICATORS

**The bot is working correctly when you see:**

✅ **Dashboard shows bot as "Running"**  
✅ **Trades are being executed** (not just status)  
✅ **Strategy signals appear in logs**  
✅ **Portfolio value changes**  
✅ **Position information updates**  

**Before fix:** Bot showed "Running" but no trading activity  
**After fix:** Complete trading pipeline execution with real market data

---

## 🎯 NEXT: CHOOSE YOUR DEPLOYMENT METHOD

**Recommended order:**
1. Try SSH connection with different ports/users
2. Try RDP if Windows server  
3. Use web panel file upload as fallback
4. Contact Strato support for access details

**Need help?** Provide:
- Server OS type (Linux/Windows)  
- SSH port and credentials
- Current bot directory path
- Any control panel access details

🚀 **Ready to deploy the fixed trading bot!**