@echo off
title Explainable Credit Card Fraud Detection
cd /d "%~dp0"

if not exist "venv\Scripts\streamlit.exe" (
    echo Virtual environment not found.
    echo Creating venv and installing dependencies...
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

echo.
echo Starting Explainable Credit Card Fraud Detection App...
echo Browser will open at http://localhost:8501
echo Keep this window open while using the app.
echo Press Ctrl+C to stop the app.
echo.

start "" "http://localhost:8501"
streamlit run app.py --server.port 8501
pause
