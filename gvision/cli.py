import os
import subprocess
from .GVISION_AUTOMATION import main

def main1():
    # Get the directory of the current module
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, "GVISION_AUTOMATION.py")
    subprocess.call(["streamlit", "run", script_path])

if __name__ == "__main__":
    main1()
