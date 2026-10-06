@echo off
title EmoSphere-XAI Public Dashboard
echo ========================================================
echo  Launching EmoSphere-XAI ML Engine...
echo ========================================================
start "EmoSphere Backend" python app.py
echo Waiting for models to initialize...
timeout /t 10 /nobreak >nul
echo ========================================================
echo  Opening Public Cloudflare Tunnel...
echo ========================================================
"C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://127.0.0.1:7860
pause
