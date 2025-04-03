import os
import sys

# Add backend directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))

# Port configuration
port = int(os.environ.get("PORT", 10000))
bind = f"0.0.0.0:{port}"

# Worker configuration
workers = 1
worker_class = 'sync'
timeout = 120
preload_app = True

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info' 