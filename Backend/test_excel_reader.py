#!/usr/bin/env python3
"""
Test script to verify Excel reading functionality
"""

import pandas as pd
import os
import sys

def test_excel_reading():
    """Test function to read and display Excel file contents"""
    excel_path = os.path.join(os.path.dirname(__file__), 'Prompts.xlsx')
    
    print(f"Testing Excel file: {excel_path}")
    print(f"File exists: {os.path.exists(excel_path)}")
    
    if not os.path.exists(excel_path):
        print("ERROR: Excel file not found!")
        return False
    
    try:
        # First, let's see what sheets are available
        excel_file = pd.ExcelFile(excel_path)
        print(f"\nAvailable sheets: {excel_file.sheet_names}")
        
        # Try to read the specific sheet
        target_sheet = "People to Test (Lineage)"
        if target_sheet in excel_file.sheet_names:
            print(f"\nReading sheet: '{target_sheet}'")
            df = pd.read_excel(excel_path, sheet_name=target_sheet)
            
            print(f"Sheet shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print("\nFirst few rows:")
            print(df.head())
            
            # Check for common genealogy-related columns
            genealogy_keywords = ['name', 'person', 'supervisor', 'advisee', 'student', 'level', 'generation']
            matching_cols = []
            for col in df.columns:
                for keyword in genealogy_keywords:
                    if keyword.lower() in col.lower():
                        matching_cols.append(col)
                        break
            
            if matching_cols:
                print(f"\nPotential genealogy columns found: {matching_cols}")
            else:
                print("\nNo obvious genealogy columns found. Will use available columns.")
            
            return True
        else:
            print(f"ERROR: Sheet '{target_sheet}' not found!")
            print("Available sheets are:", excel_file.sheet_names)
            
            # Try reading the first sheet as fallback
            if excel_file.sheet_names:
                first_sheet = excel_file.sheet_names[0]
                print(f"\nTrying first sheet '{first_sheet}' as fallback:")
                df = pd.read_excel(excel_path, sheet_name=first_sheet)
                print(f"Shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                print("\nFirst few rows:")
                print(df.head())
            
            return False
            
    except Exception as e:
        print(f"ERROR reading Excel file: {str(e)}")
        return False

if __name__ == "__main__":
    print("Excel Reader Test")
    print("=" * 50)
    success = test_excel_reading()
    
    if success:
        print("\n✅ Excel reading test PASSED")
    else:
        print("\n❌ Excel reading test FAILED")
        sys.exit(1)
