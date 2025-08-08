#!/bin/bash
# Materials Orchestrator API Service

export PYTHONPATH="/root/repo/src:$PYTHONPATH"
export MATERIALS_ORCHESTRATOR_ENV=production

cd /root/repo

echo "🚀 Starting Materials Orchestrator API..."
python3 -c "
import sys
sys.path.insert(0, 'src')

# Simple HTTP server since FastAPI might not be available
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from materials_orchestrator import AutonomousLab, MaterialsObjective, BayesianPlanner
from materials_orchestrator.quality_gates import run_all_quality_gates

class MaterialsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'name': 'Materials Orchestrator', 'status': 'running', 'version': '1.0.0'}
            self.wfile.write(json.dumps(response).encode())
        
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Quick health check
            try:
                lab = AutonomousLab()
                health = {'status': 'healthy', 'experiments_run': lab.total_experiments}
            except Exception as e:
                health = {'status': 'error', 'error': str(e)}
            
            self.wfile.write(json.dumps(health).encode())
        
        elif self.path == '/test':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Run a test experiment
            try:
                lab = AutonomousLab(planner=BayesianPlanner('band_gap'))
                objective = MaterialsObjective('band_gap', (1.2, 1.6), 'target')
                experiment = lab.run_experiment({'temperature': 150, 'time': 1})
                
                result = {
                    'status': 'success',
                    'experiment_status': experiment.status,
                    'experiment_id': experiment.id,
                    'results': experiment.results
                }
            except Exception as e:
                result = {'status': 'error', 'error': str(e)}
            
            self.wfile.write(json.dumps(result).encode())
        
        elif self.path == '/quality-gates':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Run quality gates
            try:
                gate_result = run_all_quality_gates()
                result = {
                    'status': 'success',
                    'overall_status': gate_result.overall_status.value,
                    'success_rate': gate_result.success_rate,
                    'total_tests': gate_result.total_tests,
                    'passed_tests': gate_result.passed_tests,
                    'failed_tests': gate_result.failed_tests
                }
            except Exception as e:
                result = {'status': 'error', 'error': str(e)}
            
            self.wfile.write(json.dumps(result).encode())
        
        else:
            self.send_response(404)
            self.end_headers()

server = HTTPServer(('0.0.0.0', 8000), MaterialsHandler)
print(f'🌐 Materials Orchestrator API running on port 8000')
print(f'   Health check: http://localhost:8000/health')
print(f'   Test endpoint: http://localhost:8000/test')
print(f'   Quality gates: http://localhost:8000/quality-gates')

try:
    server.serve_forever()
except KeyboardInterrupt:
    print('\n🛑 Server stopped')
    server.shutdown()
"
