import os
from pathlib import Path

# play.py resolves zoo/, logs/... relative to the working directory: app/
os.chdir(Path(__file__).resolve().parent.parent)

pytest_plugins = ['nicegui.testing.user_plugin']
