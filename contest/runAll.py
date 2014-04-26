from os import listdir
import subprocess
layoutsLocation="/u/kk8/cs343H/contest/contest/layouts/"
layouts = [ f.split(".")[0] for f in listdir(layoutsLocation)]
baselineAgent = "BaselineAgent"
ourAgent = "NotAnAgent"
for layout in layouts:
    subprocess.call(["python", "capture.py", "-r", baselineAgent, "-b", ourAgent, "-l", layout, "-t"])

