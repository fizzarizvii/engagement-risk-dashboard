@echo off
cd /d "%~dp0"
echo Starting Streamlit — your browser should open to http://localhost:8501
echo Press Ctrl+C in this window to stop the server.
python -m streamlit run app.py --server.port 8501
pause
