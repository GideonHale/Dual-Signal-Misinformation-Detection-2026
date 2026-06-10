import json

input_file = "scored_matched_claims_posts2024.jsonl"
output_file = "politics_cleaned2024.jsonl"

with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", encoding="utf-8") as outfile:

    for line in infile:
        if not line.strip():
            continue

        try:
            data = json.loads(line)

            # 1. Remove missing_source_rate if it exists
            data.pop("missing_source_rate", None)

            # 2. Change ground_truth_label to actual_score
            if "ground_truth_label" in data:
                data["actual_score"] = data.pop("ground_truth_label")

            # 3. Rename source_score to ground_truth_label
            if "source_score" in data:
                data["ground_truth_label"] = data.pop("source_score")

            # Write cleaned row
            outfile.write(json.dumps(data) + "\n")
            
        except json.JSONDecodeError:
            continue

print(f"Done. Cleaned file saved as: {output_file}")