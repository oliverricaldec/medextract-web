@echo off
REM Doble clic para abrir MedExtract en el navegador.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creando entorno e instalando dependencias, solo la primera vez...
  python -m venv .venv
  .venv\Scripts\python -m pip install -r requirements.txt
  .venv\Scripts\python -m pip install torch transformers
)
start "" http://localhost:5000
.venv\Scripts\python app.py
pause
