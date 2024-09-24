import json
import re

def process_jsonl(input_file_path, output_file_path):
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            data = json.loads(line)
            instance_id = data.get('instance_id', 'N/A')
            model_patch = data.get('model_patch', '')
            
            # Extract file name, line ranges, and function names
            patch_info = extract_patch_info(model_patch)
            
            if patch_info:
                output_file.write(f"Instance ID: {instance_id}\n")
                output_file.write(f"File: {patch_info['file']}\n")
                output_file.write("Changes:\n")
                for change in patch_info['changes']:
                    output_file.write(f"  Lines {change['line_range']}")
                    if change['function']:
                        output_file.write(f" in function {change['function']}")
                    if change['class']:
                        output_file.write(f" in class {change['class']}")
                    output_file.write("\n")
                output_file.write("-" * 40 + "\n")

def extract_patch_info(patch):
    # Regular expression to match file name
    file_pattern = r'--- a/(.*?)\n'
    file_match = re.search(file_pattern, patch)
    
    if not file_match:
        return None
    
    file_name = file_match.group(1)
    
    # Regular expression to match all line ranges and surrounding context
    range_pattern = r'@@ -(\d+),(\d+) \+(\d+),(\d+) @@(.*?)(?=\n@@|\Z)'
    range_matches = re.finditer(range_pattern, patch, re.DOTALL)
    
    changes = []
    for match in range_matches:
        start_line = int(match.group(3))  # We use the new file line numbers
        num_lines = int(match.group(4))
        end_line = start_line + num_lines - 1
        line_range = f"{start_line}-{end_line}"
        
        # Extract function name from the context
        context = match.group(5)
        function_match = re.search(r'def\s+(\w+)', context)
        class_match = re.search(r'class\s+(\w+)', context)
        function_name = function_match.group(1) if function_match else None
        class_name = class_match.group(1) if class_match else None
        
        changes.append({
            'line_range': line_range,
            'function': function_name,
            'class': class_name
        })
    
    return {
        'file': file_name,
        'changes': changes
    }

# Usage
input_file_path = '/home/wsl/SWE-bench/ground_truth_patch_Verified.jsonl'
output_file_path = 'result_patch_Verified_info_all_function_and_class.txt'
process_jsonl(input_file_path, output_file_path)
print(f"Results have been written to {output_file_path}")