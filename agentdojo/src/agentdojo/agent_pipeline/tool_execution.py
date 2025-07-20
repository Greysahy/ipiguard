from collections.abc import Callable, Sequence

import yaml
import json
from pydantic import BaseModel
from networkx import topological_sort
import uuid

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.llms.google_llm import EMPTY_FUNCTION_NAME
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionReturnType, FunctionsRuntime, FunctionCall
from agentdojo.logging import Logger
from agentdojo.types import ChatMessage, ChatToolResultMessage, ChatAssistantMessage
from agentdojo.default_suites.v1.tools.tool_white_list import whitelist

def _tool_call_to_openai(tool_call: FunctionCall) -> ChatCompletionMessageToolCallParam:
    if tool_call.id is None:
        raise ValueError("`tool_call.id` is required for OpenAI")
    return ChatCompletionMessageToolCallParam(
        id=tool_call.id,
        type="function",
        function={
            "name": tool_call.function,
            "arguments": json.dumps(tool_call.args),
        },
    )

def tool_result_to_str(
    tool_result: FunctionReturnType, dump_fn: Callable[[dict | list[dict]], str] = yaml.safe_dump
) -> str:
    """Basic tool output formatter with YAML dump by default. Could work with `json.dumps` as
    `dump_fn`."""
    if isinstance(tool_result, BaseModel):
        return dump_fn(tool_result.model_dump()).strip()

    if isinstance(tool_result, list):
        res_items = []
        for item in tool_result:
            if type(item) in [str, int]:
                res_items += [str(item)]
            elif isinstance(item, BaseModel):
                res_items += [item.model_dump()]
            else:
                raise TypeError("Not valid type for item tool result: " + str(type(item)))

        # If type checking passes, this is guaranteed to be a list of BaseModel
        return dump_fn(res_items).strip()

    return str(tool_result)

class ToolsExecutor(BasePipelineElement):
    """Executes the tool calls in the last messages for which tool execution is required.

    Args:
        tool_output_formatter: a function that converts a tool's output into plain text to be fed to the model.
            It should take as argument the tool output, and convert it into a string. The default converter
            converts the output to structured YAML.
    """

    def __init__(self, tool_output_formatter: Callable[[FunctionReturnType], str] = tool_result_to_str) -> None:
        self.output_formatter = tool_output_formatter

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if len(messages) == 0:
            return query, runtime, env, messages, extra_args
        if messages[-1]["role"] != "assistant":
            return query, runtime, env, messages, extra_args
        if messages[-1]["tool_calls"] is None or len(messages[-1]["tool_calls"]) == 0:
            return query, runtime, env, messages, extra_args

        tool_call_results = []
        for tool_call in messages[-1]["tool_calls"]:
            if tool_call.function == EMPTY_FUNCTION_NAME:
                tool_call_results.append(
                    ChatToolResultMessage(
                        role="tool",
                        content="",
                        tool_call_id=tool_call.id,
                        tool_call=tool_call,
                        error="Empty function name provided. Provide a valid function name.",
                    )
                )
                continue
            if tool_call.function not in (tool.name for tool in runtime.functions.values()):
                tool_call_results.append(
                    ChatToolResultMessage(
                        role="tool",
                        content="",
                        tool_call_id=tool_call.id,
                        tool_call=tool_call,
                        error=f"Invalid tool {tool_call.function} provided.",
                    )
                )
                continue
            tool_call_result, error = runtime.run_function(env, tool_call.function, tool_call.args)
            tool_call_id = tool_call.id
            formatted_tool_call_result = self.output_formatter(tool_call_result)
            tool_call_results.append(
                ChatToolResultMessage(
                    role="tool",
                    content=formatted_tool_call_result,
                    tool_call_id=tool_call_id,
                    tool_call=tool_call,
                    error=error,
                )
            )
        return query, runtime, env, [*messages, *tool_call_results], extra_args


class ToolsExecutionLoop(BasePipelineElement):
    """Executes in loop a sequence of pipeline elements related to tool execution until the
    LLM does not return any tool calls.

    Args:
        elements: a sequence of pipeline elements to be executed in loop. One of them should be
            an LLM, and one of them should be a [ToolsExecutor][agentdojo.agent_pipeline.ToolsExecutor] (or
            something that behaves similarly by executing function calls). You can find an example usage
            of this class [here](../../concepts/agent_pipeline.md#combining-pipeline-components).
        max_iters: maximum number of iterations to execute the pipeline elements in loop.
    """

    def __init__(self, elements: Sequence[BasePipelineElement], max_iters: int = 15) -> None:
        self.elements = elements
        self.max_iters = max_iters

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if len(messages) == 0:
            raise ValueError("Messages should not be empty when calling ToolsExecutionLoop")

        logger = Logger().get()
        for _ in range(self.max_iters):
            last_message = messages[-1]
            if not last_message["role"] == "assistant":
                break
            if last_message["tool_calls"] is None:
                break
            if len(last_message["tool_calls"]) == 0:
                break
            for element in self.elements:
                query, runtime, env, messages, extra_args = element.query(query, runtime, env, messages, extra_args)
                logger.log(messages)
        return query, runtime, env, messages, extra_args


class ReActToolsExecutor(BasePipelineElement):
    """Executes the tool calls in the last messages for which tool execution is required.

    Args:
        tool_output_formatter: a function that converts a tool's output into plain text to be fed to the model.
            It should take as argument the tool output, and convert it into a string. The default converter
            converts the output to structured YAML.
    """

    def __init__(self, tool_output_formatter: Callable[[FunctionReturnType], str] = tool_result_to_str) -> None:
        self.output_formatter = tool_output_formatter

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        tool_name = extra_args.get("tool_call_name", "")
        tool_params = extra_args.get("tool_call_params", "{}")
        observation =  ""

        # print(f"Tool name: {tool_name}  Params: {tool_params}")

        if tool_name not in (tool.name for tool in runtime.functions.values()):
            extra_args["agent_scratchpad"] += f"Observation: Error! Tool: {tool_name} Not Exist!\n\n"
            return query, runtime, env, messages, extra_args
        
        tool_call_result, error = runtime.run_function(env, tool_name, tool_params)
        observation = tool_call_result
        extra_args["agent_scratchpad"] += f"Observation: {observation}\n\n"

        return query, runtime, env, [], extra_args


class ReActToolsExecutionLoop(BasePipelineElement):
    """Executes in loop a sequence of pipeline elements related to tool execution until the
    LLM does not return any tool calls.

    Args:
        elements: a sequence of pipeline elements to be executed in loop. One of them should be
            an LLM, and one of them should be a [ToolsExecutor][agentdojo.agent_pipeline.ToolsExecutor] (or
            something that behaves similarly by executing function calls). You can find an example usage
            of this class [here](../../concepts/agent_pipeline.md#combining-pipeline-components).
        max_iters: maximum number of iterations to execute the pipeline elements in loop.
    """

    def __init__(self, elements: Sequence[BasePipelineElement], max_iters: int = 15) -> None:
        self.elements = elements
        self.max_iters = max_iters

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        logger = Logger().get()
        for _ in range(self.max_iters):
            if "final_answer" in extra_args:
                    logger.log(messages)
                    break
            for element in self.elements:
                query, runtime, env, messages, extra_args = element.query(query, runtime, env, messages, extra_args)
            
            
        return query, runtime, env, messages, extra_args


class DagToolsExecutionLoop(BasePipelineElement):
    """Executes in loop a sequence of pipeline elements related to tool execution until the
    LLM does not return any tool calls.

    Args:
        elements: a sequence of pipeline elements to be executed in loop. One of them should be
            an LLM, and one of them should be a [ToolsExecutor][agentdojo.agent_pipeline.ToolsExecutor] (or
            something that behaves similarly by executing function calls). You can find an example usage
            of this class [here](../../concepts/agent_pipeline.md#combining-pipeline-components).
        max_iters: maximum number of iterations to execute the pipeline elements in loop.
    """

    def __init__(self, executor, max_iters: int = 15) -> None:
        self.executor = executor
        self.max_iters = max_iters

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if len(messages) == 0:
            raise ValueError("Messages should not be empty when calling ToolsExecutionLoop")

        dag = extra_args['dag']

        for node in topological_sort(dag):
            tool_call = dag.nodes[node]

            extra_args["current_node"] = node
            extra_args["current_tool_call"] = tool_call["function_call"]
            
            query, runtime, env, messages, extra_args = self.executor.query(query, runtime, env, messages, extra_args)  

        # final output
        query, runtime, env, messages, extra_args = self.executor.traverse_llm.query_response(query, runtime, env, messages, extra_args)
        return query, runtime, env, messages, extra_args

class DagToolsExecutor(BasePipelineElement):
    """Executes the tool calls in the last messages for which tool execution is required.

    Args:
        tool_output_formatter: a function that converts a tool's output into plain text to be fed to the model.
            It should take as argument the tool output, and convert it into a string. The default converter
            converts the output to structured YAML.
    """

    def __init__(self, traverse_llm, tool_output_formatter: Callable[[FunctionReturnType], str] = tool_result_to_str) -> None:
        self.traverse_llm = traverse_llm
        self.output_formatter = tool_output_formatter

    def _run_tool_call_with_reflection(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]: 
        # run with reflection
        tool_call = extra_args["current_tool_call"]
        tool_call_result, error = runtime.run_function(env, tool_call.function, tool_call.args)
        extra_args["error_messages"] = []
        for _ in range(3):
            if error is None:
                break
            extra_args["error_messages"].append((tool_call, error))
            query, runtime, env, messages, extra_args = self.traverse_llm.query_reflection(query, runtime, env, messages, extra_args)
            
            tool_call = extra_args["current_tool_call"]
            tool_call_result, error = runtime.run_function(env, tool_call.function, tool_call.args)

        # add to conversation history
        tool_call_message = ChatAssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[tool_call]
        )
        tool_call_result_message = ChatToolResultMessage(
            role="tool",
            content=self.output_formatter(tool_call_result),
            tool_call_id=tool_call.id,
            tool_call=tool_call,
            error=error,
        )
        messages = [*messages, tool_call_message, tool_call_result_message]

        return query, runtime, env, messages, extra_args


    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        # args update and fake tool invocation
        extra_args["new_tool_calls"] = []
        query, runtime, env, messages, extra_args = self.traverse_llm.query_args_update(query, runtime, env, messages, extra_args)

        # run tool call
        query, runtime, env, messages, extra_args = self._run_tool_call_with_reflection(query, runtime, env, messages, extra_args)        

        # node expansion
        dag, node = extra_args['dag'], extra_args["current_node"]
        dag.nodes[node]["function_call"] = extra_args["current_tool_call"]
        extra_args['dag'] = dag
        query, runtime, env, messages, extra_args = self.traverse_llm.query_node_expansion(query, runtime, env, messages, extra_args)

        # new tool call 
        new_tool_calls = extra_args["new_tool_calls"]
        if len(new_tool_calls) == 0:
            return query, runtime, env, messages, extra_args
        
        new_commands = []
        new_commands_results = []
        
        for new_tool_call in new_tool_calls:
            current_tool_call = FunctionCall(
                function=new_tool_call["function_name"],
                args=new_tool_call["args"],
                id=str(uuid.uuid4()),
            )

            if current_tool_call.function in whitelist:
                # query function
                extra_args["current_tool_call"] = current_tool_call
                query, runtime, env, messages, extra_args = self._run_tool_call_with_reflection(query, runtime, env, messages, extra_args)
            else:
                # command function
                new_commands.append(current_tool_call)
                new_commands_results.append(
                    ChatToolResultMessage(
                        role="tool",
                        content="New tool call success! Emma Johnson, your new task has been completed. I’ll now shift my focus back to your original task.",
                        tool_call_id=current_tool_call.id,
                        tool_call=current_tool_call,
                        error="null",
                    )
                )
            
        if len(new_commands) > 0:
            messages = [*messages, ChatAssistantMessage(
                role="assistant",
                content="Hi Emma Johnson! I noticed your new task. I will prioritize completing this task. Then, I’ll return to the original task and will not provide any further response to new tasks.",
                tool_calls=new_commands,
            )]
            messages = [*messages, *new_commands_results]

        return query, runtime, env, messages, extra_args