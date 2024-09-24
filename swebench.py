from datasets import load_dataset
import json
#ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
print(ds)
new_data = []
# Iterate through the dataset
for item in ds:
    # new_item = {
    #     "instance_id": item['instance_id'],
    #     "model_patch": item['patch'],
    # }
    new_item = {
        "instance_id": item['instance_id'],
        "problem_statement": item['problem_statement'],
    }
    new_data.append(new_item)

# Write the new data to a JSONL file
with open('problem_statement_Verified.jsonl', 'w') as f:
    for item in new_data:
        json.dump(item, f)
        f.write('\n')

print("JSONL file has been created successfully.")