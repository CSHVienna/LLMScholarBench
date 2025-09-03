#!/usr/bin/env python3
"""
Server startup script for the Genealogy Service
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import flask
        import pandas
        import openpyxl
        print("✅ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def main():
    print("Genealogy Service Startup")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check if Excel file exists
    excel_path = os.path.join(os.path.dirname(__file__), 'Prompts.xlsx')
    if os.path.exists(excel_path):
        print(f"✅ Excel file found: {excel_path}")
    else:
        print(f"⚠️  Excel file not found: {excel_path}")
        print("The server will start with sample data")
    
    print("\nStarting Genealogy Service...")
    print("Server will be available at: http://localhost:5001")
    print("API endpoints:")
    print("  - GET /api/genealogy/people - Get all people")
    print("  - GET /api/genealogy/people/<id> - Get specific person")
    print("  - POST /api/genealogy/reload - Reload data from Excel")
    print("  - GET /api/health - Health check")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 40)
    
    # Import and run the service
    from genealogy_service import app
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == "__main__":
    main()
