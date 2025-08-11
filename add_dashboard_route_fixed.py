#!/usr/bin/env python3
"""
Add dashboard route to Intelligence API
"""

import os

# Read the current API file
api_file = 'run_intelligence_api.py'

# Find the line where routes are defined
insert_after = "@app.route('/api/intelligence/export/<string:data_type>')"

# New route to add
new_route = '''
@app.route('/')
@app.route('/dashboard')
def serve_dashboard():
    """Serve the JANICS FREEDOM FACTORY dashboard"""
    try:
        dashboard_path = 'janics_freedom_factory_dashboard.html'
        if os.path.exists(dashboard_path):
            with open(dashboard_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Update API base URL to use correct port
            content = content.replace('http://85.215.183.30:8080/api/intelligence', '/api/intelligence')
            return content, 200, {'Content-Type': 'text/html'}
        else:
            return "Dashboard not found. Please ensure janics_freedom_factory_dashboard.html exists.", 404
    except Exception as e:
        return f"Error loading dashboard: {str(e)}", 500
'''

print("🔧 Adding dashboard route to Intelligence API...")

# Read the file
with open(api_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find where to insert
insert_index = None
for i, line in enumerate(lines):
    if insert_after in line:
        # Find the end of this function
        j = i + 1
        while j < len(lines) and not lines[j].strip().startswith('@'):
            j += 1
        insert_index = j
        break

if insert_index:
    # Insert the new route
    lines.insert(insert_index, '\n' + new_route + '\n')
    
    # Write back
    with open(api_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("✅ Dashboard route added successfully!")
    print("📊 Dashboard will be available at:")
    print("   - http://localhost:8002/")
    print("   - http://localhost:8002/dashboard")
    print("   - http://85.215.183.30:8002/")
    print("   - http://85.215.183.30:8002/dashboard")
else:
    print("❌ Could not find insertion point!")