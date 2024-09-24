import json
import re
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_localize_output(file_path, function_name_file):
    results = {}
    function_data = load_function_name_data(function_name_file)
    
    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file, 1):
            try:
                data = json.loads(line)
                instance_id = data['instance_id']
                
                files = data['found_files']
                
                functions = []
                classes = []
                func_list = function_data[instance_id]
                for func in func_list:
                    
                    matches = re.findall(r'(?:function|class):\s*([^(\n]+)', func[0])
                    for match in matches:
                        if '.' in match:
                            class_name, func_name = match.split('.', 1)
                            classes.append(class_name.strip())
                            functions.append(func_name.strip())
                        else:
                            functions.append(match.strip())
                
                lines = []
                for edit_list in data['found_edit_locs']:
                    for item in edit_list:
                        for line in item.split('\n'):
                            match = re.search(r'line:\s*(\d+)', line)
                            if match:
                                lines.append(int(match.group(1)))
                
                results[instance_id] = {
                    'files': files,
                    'functions': list(set(functions)),
                    'classes': list(set(classes)),
                    'lines': lines
                }
            except Exception as e:
                logging.error(f"Error processing line {line_num}: {e}")
    
    return results

def load_function_name_data(file_path):
    function_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            instance_id = data['instance_id']
            function_data[instance_id] = data['model_found_locs_separated']
    return function_data


def parse_ground_truth(file_path):
    ground_truth = {}
    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file, 1):
            try:
                data = json.loads(line)
                instance_id = data['instance_id']
                patch = data['model_patch']
                
                patch_info = extract_patch_info(patch)
                
                if patch_info:
                    files = [patch_info['file']]
                    functions = set()
                    lines = set()
                    
                    for change in patch_info['changes']:
                        if change['function']:
                            functions.add(change['function'])
                        if change['class']:
                            functions.add(change['class'])
                        
                        start, end = map(int, change['line_range'].split('-'))
                        lines.update(range(start, end + 1))
                    
                    ground_truth[instance_id] = {
                        'files': files,
                        'functions': list(functions),
                        'lines': list(lines)
                    }
            except Exception as e:
                logging.error(f"Error processing ground truth line {line_num}: {e}")
    
    return ground_truth

def extract_patch_info(patch):
    file_pattern = r'--- a/(.*?)\n'
    file_match = re.search(file_pattern, patch)
    
    if not file_match:
        return None
    
    file_name = file_match.group(1)
    
    range_pattern = r'@@ -(\d+),(\d+) \+(\d+),(\d+) @@(.*?)(?=\n@@|\Z)'
    range_matches = re.finditer(range_pattern, patch, re.DOTALL)
    
    changes = []
    for match in range_matches:
        start_line = int(match.group(3))
        num_lines = int(match.group(4))
        end_line = start_line + num_lines - 1
        line_range = f"{start_line}-{end_line}"
        
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

def calculate_accuracy(localize_results, ground_truth, output_file):
    total_instances = len(ground_truth)
    correct = defaultdict(int)
    partial_correct = defaultdict(int)
    matched_instances = {
        'files': [],
        'functions': [],
        'lines': []
    }

    for instance_id, gt_data in ground_truth.items():
        if instance_id not in localize_results:
            logging.warning(f"Instance {instance_id} not found in localization results")
            continue

        loc_data = localize_results[instance_id]

        # File accuracy
        if set(gt_data['files']).issubset(set(loc_data['files'])):
            correct['files'] += 1
            matched_instances['files'].append(instance_id)
        elif set(gt_data['files']).intersection(set(loc_data['files'])):
            partial_correct['files'] += 1

        # Function and Class accuracy
        gt_functions = set(gt_data['functions'])
        loc_functions = set(loc_data['functions'])
        loc_classes = set(loc_data['classes'])
        
        if gt_functions.issubset(loc_functions.union(loc_classes)):
            correct['functions'] += 1
            matched_instances['functions'].append(instance_id)
            
            # Check line accuracy if functions are fully matched
            gt_lines = set(gt_data['lines'])
            loc_lines = set(loc_data['lines'])
            if gt_lines.intersection(loc_lines):
                correct['lines'] += 1
                matched_instances['lines'].append(instance_id)
            else:
                partial_correct['lines'] += 1
        elif gt_functions.intersection(loc_functions.union(loc_classes)):
            partial_correct['functions'] += 1

    accuracy = {
        'file': {
            'full': correct['files'] / total_instances,
            'partial': (correct['files'] + partial_correct['files']) / total_instances
        },
        'function': {
            'full': correct['functions'] / total_instances,
            'partial': (correct['functions'] + partial_correct['functions']) / total_instances
        },
        'line': {
            'full': correct['lines'] / total_instances,
            'partial': (correct['lines'] + partial_correct['lines']) / total_instances
        }
    }

    # Write matched instance IDs to file
    with open(output_file, 'w') as f:
        for category in ['files', 'functions', 'lines']:
            f.write(f"{category.upper()} MATCHES:\n")
            for instance_id in matched_instances[category]:
                f.write(f"{instance_id}\n")
            f.write("\n")

    return accuracy
# Usage
localize_file_path = '/home/wsl/AgentlessOri/Agentless/results_0822_SWE-Bench_Verified/location/loc_outputs.jsonl'
ground_truth_file_path = '/home/wsl/SWE-bench/ground_truth_patch_Verified.jsonl'
function_name_file = '/home/wsl/AgentlessOri/Agentless/results_0822_SWE-Bench_Verified/location/function_name_rethink.jsonl'
output_file_path = 'matched_instances_rethink.txt'

localize_results = parse_localize_output(localize_file_path, function_name_file)
ground_truth = parse_ground_truth(ground_truth_file_path)

accuracy = calculate_accuracy(localize_results, ground_truth, output_file_path)

print("Localization Accuracy:")
for key in ['file', 'function', 'line']:
    print(f"{key.capitalize()} Accuracy:")
    print(f"  Full Match: {accuracy[key]['full']:.2%}")
    print(f"  Partial Match: {accuracy[key]['partial']:.2%}")

print(f"\nMatched instances have been written to {output_file_path}")

# Debug information
print("\nDebug Information:")
print(f"Total instances in ground truth: {len(ground_truth)}")
print(f"Total instances in localize results: {len(localize_results)}")
print("\nSample ground truth entry:")
sample_gt = next(iter(ground_truth.values()))
print(json.dumps(sample_gt, indent=2))
print("\nSample localize result entry:")
sample_loc = next(iter(localize_results.values()))
print(json.dumps(sample_loc, indent=2))