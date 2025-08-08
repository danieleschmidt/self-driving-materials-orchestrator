#!/bin/bash
# Start Materials Orchestrator Production Services

echo "🚀 Materials Orchestrator Production Startup"
echo "============================================="

# Start API in background
echo "Starting API service..."
./start_api_simple.sh > /tmp/materials-api.log 2>&1 &
API_PID=$!

# Start dashboard in background
echo "Starting dashboard..."
./start_dashboard_simple.sh > /tmp/materials-dashboard.log 2>&1 &
DASHBOARD_PID=$!

echo "API_PID=$API_PID" > /tmp/materials-services.pid
echo "DASHBOARD_PID=$DASHBOARD_PID" >> /tmp/materials-services.pid

sleep 2

echo ""
echo "✅ Services started successfully!"
echo ""
echo "🌐 Available Services:"
echo "   • API Server: http://localhost:8000"
echo "   • Dashboard: http://localhost:8501"
echo ""
echo "🔍 Quick Tests:"
echo "   curl http://localhost:8000/health"
echo "   curl http://localhost:8000/test"
echo ""
echo "📋 Management:"
echo "   • View logs: tail -f /tmp/materials-*.log"
echo "   • Stop services: pkill -f 'Materials Orchestrator'"
echo ""
