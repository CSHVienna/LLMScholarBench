#!/usr/bin/env python3

import sys
import os
sys.path.append('LLMCaller')

from validation.validator import ResponseValidator

# Initialize validator
validator = ResponseValidator(schema_dir='LLMCaller/config/schemas')

# Test case 1: String Career Age (should fail)
test_case_1 = '''
[
    {"Name": "David J. Thouless", "Career Age": "60"},
    {"Name": "F. Duncan M. Haldane", "Career Age": "45"}
]
'''

# Test case 2: Integer Career Age (should pass)  
test_case_2 = '''
[
    {"Name": "David J. Thouless", "Career Age": 60},
    {"Name": "F. Duncan M. Haldane", "Career Age": 45}
]
'''

print("Testing seniority validation...")
print("=" * 50)

print("\nTest 1: Career Age as strings (current failing case)")
valid, message, data = validator.validate_response(test_case_1, 'seniority')
print(f"Valid: {valid}")
print(f"Message: {message}")

print("\nTest 2: Career Age as integers (expected working case)")
valid, message, data = validator.validate_response(test_case_2, 'seniority')
print(f"Valid: {valid}")
print(f"Message: {message}")