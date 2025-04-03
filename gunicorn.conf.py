import os
import sys
from pathlib import Path

# Debug prints
print(f"=== Gunicorn Configuration ===")
print(f"Current directory: {os.getcwd()}")
print(f"PORT env var: {os.environ.get('PORT')}")

# Add backend directory to Python path
backend_dir = Path(__file__).parent / 'backend'
sys.path.insert(0, str(backend_dir))
print(f"Added to path: {backend_dir}")

# Port configuration
port = int(os.environ.get("PORT", 10000))
bind = f"0.0.0.0:{port}"
print(f"Binding to: {bind}")

# Worker configuration
workers = 1
worker_class = 'sync'
timeout = 120
preload_app = True

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'debug'  # Changed to debug for more information
capture_output = True
enable_stdio_inheritance = True 