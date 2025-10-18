import json
import os
import sys
from data_loader import create_data_loaders

PER_CLASS_CSV = os.path.join('results', 'eval_test_conf', 'per_class_counts.csv')


def load_per_class_names(csv_path):
    names = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                names.append(parts[1])
    return names


def main():
    # Use defaults from create_data_loaders signature
    train_txt = os.path.join('data', 'train.txt')
    test_txt = os.path.join('data', 'test.txt')
    train_dir = os.path.join('data', 'Train')
    test_dir = os.path.join('data', 'Test')

    print('Creating data loaders (will print class lists) ...')
    try:
        train_loader, val_loader, test_loader, num_classes, class_names = create_data_loaders(
            train_dir, train_txt, test_dir, test_txt, batch_size=4, num_workers=0)
    except Exception as e:
        print('Failed to create data loaders:', e)
        sys.exit(2)

    print('\nClass names from training dataset (first 20):')
    print(class_names[:20])
    print('Total classes (train): {}'.format(len(class_names)))

    # Load per-class CSV names
    if not os.path.exists(PER_CLASS_CSV):
        print('Per-class counts CSV not found at', PER_CLASS_CSV)
        sys.exit(2)

    per_names = load_per_class_names(PER_CLASS_CSV)
    print('\nClass names from per_class_counts.csv (first 20):')
    print(per_names[:20])
    print('Total classes (per-class CSV): {}'.format(len(per_names)))

    # Compare lists
    mismatches = []
    for i, name in enumerate(per_names):
        if i >= len(class_names):
            mismatches.append((i, name, None))
        else:
            if name != class_names[i]:
                mismatches.append((i, name, class_names[i]))

    if mismatches:
        print('\nDetected mismatches between per-class CSV and train class order:')
        for idx, csv_name, train_name in mismatches[:50]:
            print(f'  idx {idx}: csv="{csv_name}" vs train="{train_name}"')
        print('\nTotal mismatches:', len(mismatches))
    else:
        print('\nNo mismatches detected; class orders match.')


if __name__ == "__main__":
    main()
