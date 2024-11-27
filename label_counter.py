from collections import Counter

label_counter = Counter()

with open('data.csv', 'r', encoding='utf-8') as file:
    lines = file.readlines()

for line in lines:
    parts = line.strip().split(',')

    if len(parts) >= 2:
        label = parts[7]
        label_counter[label] += 1

for label, count in label_counter.items():
    print(f"{label}: {count}")