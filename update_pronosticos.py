#!/usr/bin/env python3
"""
Script to update all remaining pronosticos assignments in simulacion.py
to use the shifted version for period alignment
"""

import re

# Read the file
with open("services/simulacion.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find and replace all remaining "pronosticos = dict(enumerate(fila))" patterns
# that haven't been updated yet
patterns = [
    # Pattern 1: Simple assignment
    (r'(\s+)pronosticos = dict\(enumerate\(fila\)\)(\s*\n)',
     r'\1pronosticos_original = dict(enumerate(fila))\n\1\n\1# Shift pronosticos data to align simulation periods with actual forecasts\n\1pronosticos = shift_pronosticos_data_for_simulation(pronosticos_original)\2'),
]

original_count = len(re.findall(r'pronosticos = dict\(enumerate\(fila\)\)', content))
print(f"Found {original_count} remaining pronosticos assignments to update")

# Apply replacements
for pattern, replacement in patterns:
    content, count = re.subn(pattern, replacement, content)
    if count > 0:
        print(f"Updated {count} occurrences with pattern")

# Check final count
final_count = len(re.findall(r'pronosticos = dict\(enumerate\(fila\)\)', content))
updated_count = len(re.findall(r'pronosticos_original = dict\(enumerate\(fila\)\)', content))

print(f"Remaining pronosticos assignments: {final_count}")
print(f"Updated pronosticos_original assignments: {updated_count}")

if final_count == 0:
    # Write back to file
    with open("services/simulacion.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ Successfully updated all pronosticos assignments!")
else:
    print("⚠️ Some assignments still need manual updating")