import json
import sys

def parse_jsonl(file_path, output_file_path):
    # 重定向标准输出到文件和控制台
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()  # 如果您希望立即写入文件
        def flush(self):
            for f in self.files:
                f.flush()

    with open(output_file_path, 'w') as output_file:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, output_file)

        with open(file_path, 'r') as file:
            for line in file:
                try:
                    json_obj = json.loads(line)
                    if 'instance_id' in json_obj:
                        print(json_obj['instance_id'])
                    if 'found_files' in json_obj:
                        print("found_files:")
                        print(json_obj['found_files'])
                    if 'found_related_locs' in json_obj:
                        print("found_related_locs:")
                        print(json_obj['found_related_locs'])
                    if 'found_edit_locs' in json_obj:
                        print("found_edit_locs:")
                        print(json_obj['found_edit_locs'])
                    print('-' * 40)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from line: {line}")

        # 恢复原始的标准输出
        sys.stdout = original_stdout


# 使用示例
#file_path = '/home/wsl/AgentlessOri/Agentless/results_0822_SWE-Bench_Verified/location/20case_agenless_gpt4o_localization.jsonl'  # 请替换为您的JSONL文件路径
#file_path = '/home/wsl/AgentlessOri/Agentless/results_0822_SWE-Bench_Verified/location/loc_outputs.jsonl'  # 请替换为您的JSONL文件路径
file_path = '/home/wsl/AgentlessPairwise/Agentless/results_document_string/location/loc_outputs.jsonl'  # 请替换为您的JSONL文件路径
output_file_path = 'localization_pairwise_document_string.txt'  # 输出文件的路径
parse_jsonl(file_path, output_file_path)

print(f"解析结果已保存到 {output_file_path}")