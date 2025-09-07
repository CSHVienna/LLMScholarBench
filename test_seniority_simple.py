#!/usr/bin/env python3

import json
from jsonschema import validate, ValidationError

# Load the seniority schema
with open('LLMCaller/config/schemas/seniority.json', 'r') as f:
    schema = json.load(f)

print("Seniority Schema:")
print(json.dumps(schema, indent=2))
print("\n" + "=" * 50)

# Test case 1: String Career Age (should fail)
test_data_1 = [
    {"Name": "David J. Thouless", "Career Age": "60"},
    {"Name": "F. Duncan M. Haldane", "Career Age": "45"}
]

# Test case 2: Integer Career Age (should pass)  
test_data_2 = [
    {"Name": "David J. Thouless", "Career Age": 60},
    {"Name": "F. Duncan M. Haldane", "Career Age": 45}
]

print("\nTest 1: Career Age as strings")
try:
    validate(instance=test_data_1, schema=schema)
    print("✅ Valid")
except ValidationError as e:
    print(f"❌ Invalid: {e.message}")

print("\nTest 2: Career Age as integers")
try:
    validate(instance=test_data_2, schema=schema)
    print("✅ Valid")
except ValidationError as e:
    print(f"❌ Invalid: {e.message}")