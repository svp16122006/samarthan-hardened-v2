@echo off
setlocal
if not exist .venv\Scripts\python.exe (
  echo Creating Python 3.14 virtual environment...
  py -3.14 -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
if not exist models\model.joblib python scripts\train_model.py
python -m uvicorn app.main:app --reload
