import csv
import os
from collections import defaultdict

CONF_DIR = os.path.join('results','eval_test_conf')
CM_FILE = os.path.join(CONF_DIR,'confusion_matrix.csv')
PC_FILE = os.path.join(CONF_DIR,'per_class_counts.csv')


def load_per_class(pc_file):
    rows = []
    with open(pc_file, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                'class_id': int(row['class_id']),
                'class_name': row['class_name'],
                'total': int(row['total']),
                'correct': int(row['correct']),
                'accuracy': float(row['accuracy'])
            })
    return rows


def load_confusion(cm_file):
    with open(cm_file, newline='', encoding='utf-8') as f:
        r = csv.reader(f)
        header = next(r)
        # header: empty cell then class_0..class_n
        class_cols = [h for h in header[1:]]
        cm = defaultdict(dict)
        for row in r:
            row_class = row[0]
            counts = [int(x) for x in row[1:]]
            for col_name, count in zip(class_cols, counts):
                cm[int(row_class.split('_')[1])][int(col_name.split('_')[1])] = count
    return cm


def top_confusions_for_class(cm, class_id, topk=5):
    row = cm[class_id]
    pairs = sorted(row.items(), key=lambda x: x[1], reverse=True)
    return [(tgt, cnt) for tgt,cnt in pairs if tgt != class_id][:topk]


def global_top_confusions(cm, topk=20):
    pairs = []
    for src, row in cm.items():
        for tgt, cnt in row.items():
            if src != tgt and cnt>0:
                pairs.append((src,tgt,cnt))
    pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
    return pairs[:topk]


def worst_classes(per_class_rows, min_support=4, worst_k=20):
    filtered = [r for r in per_class_rows if r['total']>=min_support]
    return sorted(filtered, key=lambda x: x['accuracy'])[:worst_k]


def main():
    pc = load_per_class(PC_FILE)
    cm = load_confusion(CM_FILE)

    print('Loaded per-class counts:', len(pc), 'classes')
    print('Loaded confusion matrix rows:', len(cm))
    print()

    worst = worst_classes(pc, min_support=4, worst_k=25)
    print('Worst classes (by accuracy, min support=4):')
    for r in worst:
        # per_class_counts.csv stores accuracy as percent (e.g. 85.7), but some tools use fraction (0.857).
        acc = r['accuracy']
        if acc <= 1.0:
            acc_pct = acc * 100.0
        else:
            acc_pct = acc
        print(f"{r['class_id']:3d} {r['class_name']:<30} support={r['total']:3d} acc={acc_pct:5.1f}%")
    print()

    print('Top global confusion pairs (src -> tgt : count):')
    for src,tgt,cnt in global_top_confusions(cm, topk=30):
        print(f"{src:3d} -> {tgt:3d} : {cnt}")
    print()

    print('Per-worst-class top confusions:')
    for r in worst[:10]:
        src = r['class_id']
        print(f"\n{src} {r['class_name']} (support={r['total']} acc={r['accuracy']*100:.1f}%) top confusions:")
        for tgt,cnt in top_confusions_for_class(cm, src, topk=6):
            tgt_name = next((x['class_name'] for x in pc if x['class_id']==tgt), str(tgt))
            print(f"  -> {tgt:3d} {tgt_name:<30} count={cnt}")

if __name__=='__main__':
    main()
