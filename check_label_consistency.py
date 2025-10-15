import os
from collections import Counter

TRAIN_TXT = 'data/train.txt'
TEST_TXT = 'data/test.txt'


def read_labels(path):
    labels = []
    if not os.path.exists(path):
        print('File not found:', path)
        return labels
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    labels.append(int(parts[1]))
                except ValueError:
                    # non-integer label, treat as string
                    labels.append(' '.join(parts[1:]))
    return labels


def summarize(labels):
    c = Counter(labels)
    total = sum(c.values())
    unique = len(c)
    most_common = c.most_common(10)
    return total, unique, most_common, c


if __name__ == '__main__':
    train_labels = read_labels(TRAIN_TXT)
    test_labels = read_labels(TEST_TXT)

    t_total, t_unique, t_common, t_counts = summarize(train_labels)
    s_total, s_unique, s_common, s_counts = summarize(test_labels)

    print('Train: {} samples, {} unique labels'.format(t_total, t_unique))
    print('Test:  {} samples, {} unique labels'.format(s_total, s_unique))
    print('\nTop 10 train label counts:')
    for lab, cnt in t_common:
        print('  {}: {}'.format(lab, cnt))

    print('\nTop 10 test label counts:')
    for lab, cnt in s_common:
        print('  {}: {}'.format(lab, cnt))

    # Find labels in test not in train
    train_set = set(t_counts.keys())
    test_set = set(s_counts.keys())
    missing_in_train = sorted(list(test_set - train_set))
    missing_in_test = sorted(list(train_set - test_set))

    print('\nLabels in test but not in train (sample):', missing_in_train[:20])
    print('Labels in train but not in test (sample):', missing_in_test[:20])

    # Compare counts for overlapping labels
    overlap = sorted(list(train_set & test_set))
    diffs = []
    for lab in overlap:
        diffs.append((lab, t_counts[lab], s_counts[lab]))
    diffs_sorted = sorted(diffs, key=lambda x: abs(x[1]-x[2]), reverse=True)[:20]
    print('\nTop 20 label count differences (label, train_count, test_count):')
    for item in diffs_sorted:
        print('  {}'.format(item))

    # Simple sanity: check if labels look numeric and in similar ranges
    try:
        if t_unique > 0 and s_unique > 0:
            print('\nMin/Max train label:', min(train_labels), max(train_labels))
            print('Min/Max test label: ', min(test_labels), max(test_labels))
    except Exception:
        pass

    print('\nDone. If you see labels in test that are not present in train, that indicates a label mapping issue.')
