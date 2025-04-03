import os
import sys
from blood_grp import app

def create_app():
    print("=== Creating Flask Application ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print(f"PORT env var: {os.environ.get('PORT')}")
    return app

application = create_app() 