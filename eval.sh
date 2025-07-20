#!/bin/bash
export OPENAI_API_KEY='<your_api_key>'
export OPENAI_BASE_URL='<your_base_url>'

agent_model="gpt-4o-mini-2024-07-18"
attack_name="important_instructions"
defense_name="ipiguard"
suite_name="travel"
mode="benign"

output_dir="evaluation_results/$(echo $suite_name | tr '/' '_')/$(echo $agent_model | tr '/' '_')/$(echo $mode | tr '/' '_')_$(echo $attack_name | tr ' ' '_')_$(echo $defense_name | tr ' ' '_')_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$output_dir"

python3 run/eval.py \
    --suite_name "$suite_name" \
    --agent_model "$agent_model" \
    --attack_name "$attack_name" \
    --defense_name "$defense_name" \
    --output_dir "$output_dir" \
    --mode "$mode" \
    --uid 0 \
    --iid 0 \