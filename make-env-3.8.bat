start /MAX cmd /c "cls && title Prepare Environment && py -3.8 -m venv venv && cd venv/Scripts && activate && cd .. && cd .. && pip install -r requirements.txt && deactivate && timeout /t 5 /nobreak"
