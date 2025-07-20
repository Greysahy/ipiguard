import json
import os
import heapq
import re
from agentdojo.functions_runtime import FunctionCall

def repair_prompt(prompt):
    return prompt.replace('"', "'")

def extract_attack_prompt(prompt):
    pattern = r'<REWRITTEN PROMPT>(.*?)</REWRITTEN PROMPT>'
    match = re.search(pattern, prompt, re.DOTALL)
    
    if match:
        rewritten_prompt = match.group(1).strip()
        return rewritten_prompt
    else:
        return None

class FunctionCallEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, FunctionCall):
            return obj.model_dump()
        return super().default(obj)

def save_result(result, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, "results.jsonl")
    with open(result_file, "a") as f:
        json.dump(result, f, cls=FunctionCallEncoder, indent=4)
        f.write("\n")


def sanitize_prompt(prompt):
    yaml_special_chars = r'[:#{}[\],%&*!|>?\'\"]'
    sanitized_string = re.sub(yaml_special_chars, '', prompt)
    sanitized_string = re.sub(r'\s+', ' ', sanitized_string).strip()
    
    return sanitized_string

def save_prefix(heap, prefix, reward, heap_size):
    if len(heap) < heap_size:
        heapq.heappush(heap, (reward, prefix))
    else:
        if reward >= heap[0][0]:
            heapq.heappop(heap)
            heapq.heappush(heap, (reward, prefix))

def get_tool_docs(tools):
    tools_docs = ""
    for index, tool in enumerate(tools, start=1):
        tool_dict = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters.model_json_schema(),
        }
        tools_docs += f"<function-{index}>\n"
        tools_docs += json.dumps(tool_dict, indent=4)
        tools_docs += f"\n</function-{index}>\n\n"
    
    return tools_docs