#!/usr/bin/env python3
import sys
from collections import defaultdict

# To store counts and predictions
class_counts = defaultdict(int)
feature_class_counts = defaultdict(lambda: defaultdict(int))
total_instances = 0
correct_predictions = 0

for line in sys.stdin:
    key, value = line.strip().split('\t')

    if key.startswith("Feature"):
        feature, label = key.rsplit('_Class_', 1)
        feature_class_counts[feature][label] += int(value)
    elif key.startswith("Class"):
        class_counts[value] += int(value)
    elif key == "Actual_Label":
        actual_label = value
        total_instances += 1

        # Naive Bayes Prediction logic
        max_likelihood = None
        prediction = None
        for label in class_counts:
            likelihood = class_counts[label]
            for feature, feature_count in feature_class_counts.items():
                if feature in feature_count:
                    likelihood *= feature_count[label]

            if max_likelihood is None or likelihood > max_likelihood:
                max_likelihood = likelihood
                prediction = label

        # Check if prediction matches actual label
        if prediction == actual_label:
            correct_predictions += 1

# Calculate accuracy
print(f"Accuracy\t{correct_predictions / total_instances * 100:.2f}%")
