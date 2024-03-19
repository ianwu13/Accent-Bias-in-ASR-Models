import os
import json

def md_to_json(md_file):
    data = {}
    current_header = None

    with open(md_file, 'r') as f:
        for line in f:
            if '###' in line:
                current_header = line.replace("###", "").strip().lower()
                if current_header == "other":
                    current_header = None
                    continue
            elif line.startswith("    ") and current_header:
                data[line.strip()] = current_header

    return data

# Provide the path to your markdown file here
md_file_path = "data/common_voice_16/accents.md"
output_json_file = "output.json"

json_data = md_to_json(md_file_path)

with open(output_json_file, 'w') as outfile:
    json.dump(json_data, outfile, indent=4)