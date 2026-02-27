#!/bin/bash

# Navigate to the project directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists (optional)
# if [ -d "venv" ]; then
#     source venv/bin/activate
# fi

# Run the Python application
python3 src/main.py
