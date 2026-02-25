#!/bin/bash

# Rayx-Claw Start Script
# Usage: ./start.sh [port]
#   - No args: runs on default port 8000
#   - With port: runs on specified port (e.g., ./start.sh 3000)

cd "$(dirname "$0")"

# Get port from argument or use default
PORT=${1:-8000}

echo "🚀 Starting Rayx-Claw on port $PORT..."

# Install dependencies if needed
if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
    echo "📦 Installing Python dependencies..."
    pip3 install -r requirements.txt
fi

# Start the server
echo "✅ Server running at http://localhost:$PORT"
python3 -m uvicorn server:app --host 0.0.0.0 --port "$PORT"
