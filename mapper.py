#!/usr/bin/env python3
import sys

def emit(key, value):
    print(f"{key}\t{value}")

for line in sys.stdin:
    tokens = line.strip().split(',')
    label = tokens[-1]  # Assuming the last column is the label (Outcome)

    # Emit features along with the label
    for i in range(len(tokens) - 1):
        emit(f"Feature_{i}_Value_{tokens[i]}", label)

    # Emit the actual label for accuracy calculation
    emit(f"Actual_Label", label)
