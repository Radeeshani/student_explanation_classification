#!/bin/bash
# Script to start Jupyter notebook with the virtual environment

echo "Starting Jupyter notebook with virtual environment..."
source venv/bin/activate
echo "Virtual environment activated!"
echo "Starting Jupyter notebook..."
echo "You can now access your notebooks at http://localhost:8888"
echo ""
echo "To stop Jupyter, press Ctrl+C in the terminal"
jupyter notebook 