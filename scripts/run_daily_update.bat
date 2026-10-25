@echo off
cd /d "%~dp0.."
python scripts\update_prices.py
if errorlevel 1 exit /b 1
python scripts\update_news.py
exit /b %errorlevel%

:: Windows Task Scheduler: run once in a Command Prompt to register the weekday 19:15 job
:: schtasks /Create /TN "ml_alpha_daily_update" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 19:15 /TR "\"C:\Users\jayan\Study Materials\ml_alpha_portfolio\ml-alpha-portfolio-model\scripts\run_daily_update.bat\""
