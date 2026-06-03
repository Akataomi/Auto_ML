@echo off
REM Script to start MLFlow server locally (without Docker)
REM For development and testing only

echo Starting MLFlow server (local mode)...
echo Tracking URI: file://%CD%\mlruns
echo Open in browser: http://localhost:5000
echo.
echo WARNING: This is for local development only!
echo For production use: docker compose up -d
echo.

mlflow server ^
    --host 127.0.0.1 ^
    --port 5000 ^
    --backend-store-uri file://%CD%\mlruns ^
    --default-artifact-root file://%CD%\mlruns
