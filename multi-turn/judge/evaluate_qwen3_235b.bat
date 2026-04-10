@echo off
REM Evaluation script using Qwen3-235B model
REM Please modify the following variables according to your API configuration

set JUDGE_API_KEY=your-api-key
set JUDGE_BASE_URL=https://your-api-url/v1
set JUDGE_MODEL_NAME=qwen3-235b
set OUTPUT_DIR=./output

REM Change to the script's directory (multi-turn directory)
cd /d "%~dp0\.."

REM Run the evaluation script
python -m judge.evaluate ^
    --output-dir %OUTPUT_DIR% ^
    --judge-api-key "%JUDGE_API_KEY%" ^
    --judge-base-url "%JUDGE_BASE_URL%" ^
    --judge-model-name "%JUDGE_MODEL_NAME%"

pause
