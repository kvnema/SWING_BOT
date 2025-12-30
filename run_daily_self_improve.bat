@echo off
cd /d C:\Users\K01340\SWING_BOT_GIT\SWING_BOT
call .venv\Scripts\activate.bat
python -c "import sys; sys.path.insert(0, 'src'); from scripts.daily_self_improve import main; main()"