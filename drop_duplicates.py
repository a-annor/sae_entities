import json

def drop_duplicates_jsonl(input_path, output_path):
    seen = set()
    unique_entries = []

    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            entry = json.loads(line)
            entry_str = json.dumps(entry, sort_keys=True)
            if entry_str not in seen:
                seen.add(entry_str)
                unique_entries.append(entry)

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for entry in unique_entries:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')

# Example usage
drop_duplicates_jsonl('data/final_result/Gender_completion_sentiment_judged_final.jsonl', 'data/final_result/new_Gender_completion_sentiment_judged_final.jsonl')
