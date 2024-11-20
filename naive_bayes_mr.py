from mrjob.job import MRJob
import math

class NaiveBayesMR(MRJob):

    def mapper(self, _, line):
        fields = line.strip().split(',')
        if fields[0] != "Pregnancies":  # Skip header row
            label = fields[-1]
            features = fields[:-1]  # All columns except the label
            yield "data", (features, label)

    def reducer(self, key, values):
        from collections import defaultdict
        import numpy as np
        
        # Split data into training/testing
        data = list(values)
        if not data:
            return
        np.random.shuffle(data)
        train_size = int(0.7 * len(data))
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        # Train Naive Bayes
        counts = defaultdict(lambda: defaultdict(int))
        label_counts = defaultdict(int)
        feature_totals = defaultdict(int)
        
        for features, label in train_data:
            label_counts[label] += 1
            for i, feature in enumerate(features):
                counts[label][(i, feature)] += 1
                feature_totals[i] += 1
        
        # Classify test data and calculate accuracy
        correct = 0
        for features, true_label in test_data:
            probs = {}
            for label in label_counts:
                prob = math.log(label_counts[label] / len(train_data))
                for i, feature in enumerate(features):
                    numerator = counts[label][(i,feature)] + 1
                    denominator = label_counts[label] + feature_totals[i]
                    prob += math.log(numerator/denominator)
                probs[label] = prob
            pred_label = max(probs, key=probs.get)
            if pred_label == true_label:
                correct += 1
        
        accuracy = correct / len(test_data) if test_data else 0
        yield "accuracy", accuracy

if __name__ == '__main__':
    NaiveBayesMR.run()
