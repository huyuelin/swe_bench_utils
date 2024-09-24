import json
import re

def parse_result_patch_path(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    instances = content.split('----------------------------------------')
    result = []

    for instance in instances:
        if not instance.strip():
            continue

        instance_data = {}
        lines = instance.strip().split('\n')

        # Extract instance_id
        instance_id_match = re.search(r'Instance ID:\s*(.+)', lines[0])
        if instance_id_match:
            instance_data['instance_id'] = instance_id_match.group(1).strip()

        # Extract found_files and found_edit_locs
        found_files = []
        found_edit_locs = []
        current_file = None

        for line in lines[1:]:
            file_match = re.match(r'File:\s*(.+)', line)
            if file_match:
                current_file = file_match.group(1)
                found_files.append(current_file)
            elif line.startswith('Changes:'):
                continue
            elif current_file:
                edit_loc = re.search(r'Lines\s+(\d+)-(\d+)(?:\s+in\s+function\s+(.+))?', line)
                if edit_loc:
                    start_line, end_line, function_name = edit_loc.groups()
                    edit_info = f"function: {function_name if function_name else '*'}\n"
                    edit_info += f"line: {start_line}"
                    found_edit_locs.append([edit_info])

        instance_data['found_files'] = found_files
        instance_data['found_edit_locs'] = found_edit_locs

        result.append(instance_data)

    return result

def write_jsonl(data, output_file):
    with open(output_file, 'w') as file:
        for item in data:
            json.dump(item, file)
            file.write('\n')

# Main execution
input_file = '/home/wsl/SWE-bench/result_patch_info_all.txt'
output_file = 'result_groundtruth_patch_loc.jsonl'

parsed_data = parse_result_patch_path(input_file)
write_jsonl(parsed_data, output_file)

print(f"JSONL file '{output_file}' has been created successfully.")