import os

def read_instances(file_path):
    instances = {'files': set(), 'functions': set(), 'lines': set()}
    current_category = None
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line in ['FILES MATCHES:', 'FUNCTIONS MATCHES:', 'LINES MATCHES:']:
                current_category = line.split()[0].lower()
            elif line and current_category:
                instances[current_category].add(line)
    
    return instances

def compare_instances(ori_instances, new_instances):
    additional = {category: new_instances[category] - ori_instances[category] 
                  for category in ori_instances}
    missing = {category: ori_instances[category] - new_instances[category] 
               for category in ori_instances}
    return additional, missing

def print_results(file_name, additional, missing):
    print(f"\nResults for {file_name}:")
    for category in ['files', 'functions', 'lines']:
        print(f"  {category.capitalize()}:")
        print(f"    Additional: {len(additional[category])}")
        print(f"    Missing: {len(missing[category])}")

def write_results(output_file, file_name, additional, missing):
    with open(output_file, 'a') as f:
        f.write(f"\nResults for {file_name}:\n")
        for category in ['files', 'functions', 'lines']:
            f.write(f"  {category.capitalize()}:\n")
            f.write(f"    Additional ({len(additional[category])}):\n")
            for instance in additional[category]:
                f.write(f"      {instance}\n")
            f.write(f"    Missing ({len(missing[category])}):\n")
            for instance in missing[category]:
                f.write(f"      {instance}\n")

def main():
    ori_file = 'matched_instances_ori.txt'
    compare_files = [
        'matched_instances_rethink.txt',
        'matched_instances_pairwise.txt',
        'matched_instances_pairwise_document_string.txt'
    ]
    output_file = 'comparison_results.txt'

    if os.path.exists(output_file):
        os.remove(output_file)

    ori_instances = read_instances(ori_file)

    for file in compare_files:
        new_instances = read_instances(file)
        additional, missing = compare_instances(ori_instances, new_instances)
        print_results(file, additional, missing)
        write_results(output_file, file, additional, missing)

    print(f"\nDetailed results have been written to {output_file}")

if __name__ == "__main__":
    main()