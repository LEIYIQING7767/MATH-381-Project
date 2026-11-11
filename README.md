# MATH-381-Project
The code (.py file) is supposed to work with the survey result (.txt file) and the .csv files are the output

To run the code on codespace, paste this in terminal: 
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install ortools==9.15.6755
  python mathily_er_scheduler.py survey.txt --out results

To run the code locally, make sure the file path is right, edit and paste this in terminal:
  cd "$env:USERPROFILE\Downloads\MathCamp"
  $pythonExe = "C:\Users\ylray\AppData\Local\Python\pythoncore-3.14-64\python.exe"
  & $pythonExe -m pip install ortools==9.15.6755
  & $pythonExe mathily_er_scheduler.py survey.txt --out results
