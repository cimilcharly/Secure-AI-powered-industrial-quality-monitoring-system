import csv

csv_file = 'c:/Users/HP/Desktop/Dress Defect/src/data_generation/prompts.csv'
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    print('--- STITCH PROMPT EXAMPLES ---')
    count = 0
    for row in reader:
        if row['class'] == 'stitch':
            print(f"\nSubtype: {row['subtype']} | Position: {row['position']}")
            print(f"Prompt: {row['full_prompt']}")
            count += 1
            if count >= 2:
                break
