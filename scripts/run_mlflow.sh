#!/bin/bash
# Script to start MLFlow server locally (without Docker)
# For development and testing only

echo "Starting MLFlow server (local mode)..."
echo "Tracking URI: file://$(pwd)/mlruns"
echo "Open in browser: http://localhost:5000"
echo ""
echo "This is for local development only!"
echo "For production use: docker compose up -d"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

mlflow server \
    --host 127.0.0.1 \
    --port 5000 \
    --backend-store-uri file://$(pwd)/mlruns \
    --default-artifact-root file://$(pwd)/mlruns
