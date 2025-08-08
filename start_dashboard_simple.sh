#!/bin/bash
# Materials Orchestrator Dashboard Service

export PYTHONPATH="/root/repo/src:$PYTHONPATH"
export MATERIALS_ORCHESTRATOR_ENV=production

cd /root/repo

echo "🖥️ Starting Materials Orchestrator Dashboard..."

# Check if streamlit is available
python3 -c "import streamlit" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Streamlit available, starting dashboard..."
    python3 -m streamlit run src/materials_orchestrator/dashboard/app.py \
        --server.port 8501 \
        --server.address 0.0.0.0 \
        --server.headless true
else
    echo "⚠️ Streamlit not available, creating simple dashboard..."
    python3 -c "
import sys
sys.path.insert(0, 'src')
from http.server import HTTPServer, BaseHTTPRequestHandler
from materials_orchestrator import AutonomousLab, MaterialsObjective, BayesianPlanner
import json

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        html = \"\"\"<!DOCTYPE html>
<html>
<head><title>Materials Orchestrator Dashboard</title></head>
<body>
    <h1>🔬 Materials Orchestrator Dashboard</h1>
    <h2>System Status: Operational</h2>
    <p>Simple dashboard - Install streamlit for full interface</p>
    <p><a href='http://localhost:8000/health'>API Health Check</a></p>
    <p><a href='http://localhost:8000/test'>Run Test Experiment</a></p>
</body>
</html>\"\"\"
        self.wfile.write(html.encode())

server = HTTPServer(('0.0.0.0', 8501), DashboardHandler)
print(f'🌐 Simple dashboard running on port 8501')
server.serve_forever()
"
fi
