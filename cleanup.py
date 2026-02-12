import json

# Input and output file paths
input_file = "train_data_status.json"
output_file = "clean_train_data_status.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

seen_texts = set()
deduplicated_data = []

for entry in data:
    text = entry[0]  # The sentence part

    if text not in seen_texts:
        seen_texts.add(text)
        deduplicated_data.append(entry)

# Write cleaned data to new file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(deduplicated_data, f, indent=2, ensure_ascii=False)

print(f"Done! {len(data) - len(deduplicated_data)} duplicates removed.")
