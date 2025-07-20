from dotenv import load_dotenv
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import Any, Optional, Union, List
import json
import os
import time
from cohere.core.api_error import ApiError
from google.api_core.exceptions import InternalServerError
from openai import BadRequestError

from agentdojo.logging import Logger, TraceLogger
from agentdojo.task_suite.task_suite import TaskSuite
from agentdojo.agent_pipeline.agent_pipeline import DEFENSES, AgentPipeline, PipelineConfig
from agentdojo.attacks.attack_registry import ATTACKS, load_attack
from agentdojo.benchmark import SuiteResults, benchmark_suite_with_injections, benchmark_suite_without_injections
from agentdojo.logging import OutputLogger
from agentdojo.models import ModelsEnum
from agentdojo.task_suite.load_suites import get_suite, get_suites
from agentdojo.task_suite.task_suite import TaskSuite
from agentdojo.attacks.agentic_attacks import AgenticAttack
from agentdojo.functions_runtime import FunctionCall
from data_module import initialize_dataset
from tqdm import tqdm
from utils import save_result

class AgentTask:

    @staticmethod
    def benign_run(agent_pipeline, suite, user_task):
        try:
            utility, security, messages, args = suite.run_task_with_pipeline_msg(
                agent_pipeline, user_task, injection_task=None, injections={}, extra_args={"input_tokens": 0, "output_tokens": 0}
            )
        except BadRequestError as e:
            if e.code == "context_length_exceeded" or e.param == "max_tokens":
                print(
                    f"Skipping task '{user_task.ID}' with '{injection_task.ID}' due to context_length_exceeded: {e}"
                )
                utility = False
                security = True
            else:
                raise e
        except ApiError as e:
            if "internal server error" in str(e):
                print(
                    f"Skipping task '{user_task.ID}' with '{injection_task.ID}' because of internal server error: {e}"
                )
                utility = False
                security = True
            else:
                raise e
        except InternalServerError as e:
                    print(
                        f"Skipping task '{user_task.ID}' with '{injection_task.ID}' because of internal server error: {e}"
                    )
                    utility = False
                    security = True

        return 1 - int(security), int(utility), messages

    @staticmethod
    def run(agent_pipeline, suite, attacks, user_task, injection_task, extra_args):
        try:
            utility, security, messages, args = suite.run_task_with_pipeline_msg(
                agent_pipeline, user_task, injection_task, attacks, extra_args={"input_tokens": 0, "output_tokens": 0}
            )
        except BadRequestError as e:
            if e.code == "context_length_exceeded" or e.param == "max_tokens":
                print(
                    f"Skipping task '{user_task.ID}' with '{injection_task.ID}' due to context_length_exceeded: {e}"
                )
                utility = False
                security = True
                messages = []
                args = {"input_tokens": 0, "output_tokens": 0}
            else:
                utility = False
                security = True
                messages = []
                args = {"input_tokens": 0, "output_tokens": 0}
        except ApiError as e:
            if "internal server error" in str(e):
                print(
                    f"Skipping task '{user_task.ID}' with '{injection_task.ID}' because of internal server error: {e}"
                )
                utility = False
                security = True
                messages = []
                args = {"input_tokens": 0, "output_tokens": 0}
            else:
                raise e
        except InternalServerError as e:
                    print(
                        f"Skipping task '{user_task.ID}' with '{injection_task.ID}' because of internal server error: {e}"
                    )
                    utility = False
                    security = True
                    messages = []
                    args = {"input_tokens": 0, "output_tokens": 0}

        return int(security), int(utility), messages

@dataclass
class ScriptArguments:
    output_dir: Optional[str] = field(
        default="eval_logs/", metadata={"help": "directory to save the eval log"}
    )
    # agent parameters
    benchmark_version: Optional[str] = field(
        default="v1.1.2", metadata={"help": "the benchmark version"}
    )
    suite_name: Optional[List[str]] = field(
        default_factory=lambda: ["workspace","slack", "travel", "banking"], 
        metadata={"help": "the suite name, can be banking, workspace, slack or travel. Multiple values can be provided."}
    )
    attack_name: Optional[str] = field(
        default="important_instructions", metadata={"help": "the name of attack method"}
    )
    defense_name: Optional[str] = field(
        default="ipiguard", metadata={"help": "the name of defense method"}
    )
    agent_model: Optional[str] = field(
        default="gpt-3.5-turbo-0125", metadata={"help": "the name of the model used for the agent's brain"}
    )
    mode: Optional[str] = field(
        default="under_attack", metadata={"help": " 'benign' or 'under_attack' "}
    )
    uid: Optional[int] = field(
        default="0", metadata={"help": " resume from user task "}
    )
    iid: Optional[int] = field(
        default="0", metadata={"help": " resume from injection task "}
    )

def benign_eval(script_args, agent_pipeline, suite, attacker, agent_test_dataset):
    sum = 0
    security = 0
    useful = 0
    for user_task_id in tqdm(agent_test_dataset, desc="Evaluating on the test dataset"):
        if user_task_id < script_args.uid:
            continue
        # user_task_to_run = suite.get_user_task_by_id(f"user_task_{1}")
        user_task_to_run = suite.get_user_task_by_id(f"user_task_{user_task_id}")
        task_reward, utility, messages = AgentTask.benign_run(
                agent_pipeline, suite, user_task_to_run)
        
        save_result(
            {
                "user_task_id": user_task_id,
                "security": task_reward,
                "utility": utility,
                "messages": messages
            },
            script_args.output_dir
        )
        sum += 1
        security += task_reward
        useful += utility
        # break
    asr = security * 100 / sum  
    ability = useful * 100 / sum

    return security, useful, sum, asr, ability



def eval(script_args, agent_pipeline, suite, attacker, agent_test_dataset):
    sum = 0
    security = 0
    useful = 0

    for user_task_id, injection_task_id in tqdm(agent_test_dataset, desc="Evaluating on the test dataset"):
        if user_task_id < script_args.uid:
            continue
        if user_task_id == script_args.uid and injection_task_id < script_args.iid:
            continue
        
        user_task_to_run = suite.get_user_task_by_id(f"user_task_{user_task_id}")
        injection_task_to_run = suite.get_injection_task_by_id(f"injection_task_{injection_task_id}")
        attacks = attacker.attack(user_task_to_run, injection_task_to_run)

        task_reward, utility, history = AgentTask.run(
                agent_pipeline, suite, attacks, user_task_to_run, injection_task_to_run , extra_args={}
                )
        
        save_result(
            {
                "user_task_id": user_task_id,
                "injection_task_id": injection_task_id,
                "security": task_reward,
                "utility": utility,
                "messages": history,
            },
            script_args.output_dir
        )
        sum += 1
        security += task_reward
        useful += utility

    asr = security * 100 / sum  
    ability = useful * 100 / sum

    return security, useful, sum, asr, ability

if __name__ == '__main__':
    load_dotenv()
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    os.makedirs(script_args.output_dir, exist_ok=True)

    save_result({"args": vars(script_args)}, script_args.output_dir)
    
    if len(script_args.suite_name) == 1 and script_args.suite_name[0] == "all":
        script_args.suite_name = ["slack", "banking", "travel", "workspace"]
    
    success = 0
    attack = 0
    all = 0
    for suite_n in tqdm(script_args.suite_name):
        suite = get_suite(script_args.benchmark_version, suite_n)
        model = ModelsEnum(script_args.agent_model)

        # Initialization
        if script_args.defense_name == "None":
            pipeline = AgentPipeline.from_config(
                PipelineConfig(
                    llm=model, defense=None, system_message_name=None, system_message=None
                )
            )
        else:
            pipeline = AgentPipeline.from_config(
                PipelineConfig(
                    llm=model, defense=script_args.defense_name, system_message_name=None, system_message=None
                )
            )
        
        attacker = load_attack(script_args.attack_name, suite, pipeline)
    
        
        if script_args.mode == "benign":
            dataset = initialize_dataset(suite_n, benign=True)
            security, useful, sum, suite_asr, ability = benign_eval(script_args, pipeline, suite, attacker, dataset) 
        elif script_args.mode == "under_attack":
            dataset = initialize_dataset(suite_n)
            security, useful, sum, suite_asr, ability = eval(script_args, pipeline, suite, attacker, dataset)
        else:
            raise ValueError(f"Unknown mode: {script_args.mode}")
        
        success += useful
        attack += security
        all += sum
        print(f"Suite: {suite_n}, ASR: {suite_asr} %, Utility: {ability} %")
        save_result({"Suite": suite_n, "ASR": suite_asr, "Utility": ability}, script_args.output_dir)
        abi = success * 100 / all   
        asr = attack * 100 / all
        save_result({"ASR": asr, "Utility": abi}, script_args.output_dir)
        print(f"Overall ASR: {asr} %, Overall Utility: {abi} %")