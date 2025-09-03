#!/bin/bash

# Genealogy API Startup Script

echo "🚀 Starting Genealogy API Server..."
echo "=================================="

# Navigate to backend directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if Excel file exists
if [ ! -f "Prompts.xlsx" ]; then
    echo "⚠️  Excel file 'Prompts.xlsx' not found!"
    echo "   The server will start with sample data"
fi

echo "✅ Virtual environment activated"
echo "✅ Starting Flask server on http://localhost:5001"
echo ""
echo "📋 Available API endpoints:"
echo "   - GET  /api/genealogy/people      - Get all people"
echo "   - GET  /api/genealogy/people/<id> - Get specific person"
echo "   - POST /api/genealogy/reload      - Reload Excel data"
echo "   - GET  /api/health                - Health check"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=================================="

# Start the server
python genealogy_service.py
