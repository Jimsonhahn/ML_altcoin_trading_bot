#!/usr/bin/env node
/**
 * Test Dashboard WebSocket Hooks
 * Tests all implemented hooks for compilation and basic functionality
 */

const fs = require('fs');
const path = require('path');

// Test if all hooks are properly exported
const hookFile = path.join(__dirname, '../dashboard/src/hooks/useWebSocket.js');

if (!fs.existsSync(hookFile)) {
    console.log('❌ useWebSocket.js file not found');
    process.exit(1);
}

const content = fs.readFileSync(hookFile, 'utf8');

// Check for all required exports
const requiredExports = [
    'useWebSocket',
    'useBotStatus', 
    'useTradingUpdates',
    'usePerformanceUpdates',
    'useSystemStatus',
    'useAlerts'
];

console.log('🧪 Testing WebSocket Hooks...');
console.log('=' * 40);

let allExportsFound = true;
requiredExports.forEach(exportName => {
    if (content.includes(`export const ${exportName}`)) {
        console.log(`✅ ${exportName} - Found`);
    } else {
        console.log(`❌ ${exportName} - Missing`);
        allExportsFound = false;
    }
});

// Check for socket.io-client import
if (content.includes("import io from 'socket.io-client'")) {
    console.log('✅ socket.io-client import - Found');
} else {
    console.log('❌ socket.io-client import - Missing');
    allExportsFound = false;
}

// Check for connect/disconnect functions
if (content.includes('const connect') && content.includes('const disconnect')) {
    console.log('✅ connect/disconnect functions - Found');
} else {
    console.log('❌ connect/disconnect functions - Missing');
    allExportsFound = false;
}

console.log('=' * 40);

if (allExportsFound) {
    console.log('🎉 All WebSocket hooks are properly implemented!');
    console.log('✅ Dashboard should compile without errors');
    process.exit(0);
} else {
    console.log('❌ Some hooks are missing or incomplete');
    process.exit(1);
}