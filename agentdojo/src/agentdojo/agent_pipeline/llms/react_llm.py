import json
import re
from collections.abc import Sequence

import openai
from openai._types import NOT_GIVEN
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
from openai.types.shared_params import FunctionDefinition
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_random_exponential

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, Function, FunctionCall, FunctionsRuntime
from agentdojo.types import ChatAssistantMessage, ChatMessage, ChatUserMessage


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


def _message_to_openai(message: ChatMessage) -> ChatCompletionMessageParam:
    match message["role"]:
        case "system":
            return ChatCompletionSystemMessageParam(role="system", content=message["content"])
        case "user":
            return ChatCompletionUserMessageParam(role="user", content=message["content"])
        case "assistant":
            if message["tool_calls"] is not None and len(message["tool_calls"]) > 0:
                tool_calls = [_tool_call_to_openai(tool_call) for tool_call in message["tool_calls"]]
                return ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=message["content"],
                    tool_calls=tool_calls,
                )
            return ChatCompletionAssistantMessageParam(
                role="assistant",
                content=message["content"],
            )
        case "tool":
            # print(message)
            if message["tool_call_id"] is None:
                raise ValueError("`tool_call_id` should be specified for OpenAI.")
            return ChatCompletionToolMessageParam(
                content=message["error"] or message["content"],
                tool_call_id=message["tool_call_id"],
                role="tool",
                name=message["tool_call"].function,  # type: ignore -- this is actually used, and is important!
            )
        case _:
            raise ValueError(f"Invalid message type: {message}")


def _openai_to_tool_call(tool_call: ChatCompletionMessageToolCall) -> FunctionCall:
    arguments=json.loads(tool_call.function.arguments)
    if isinstance(arguments, str):
        arguments=json.loads(arguments)
    return FunctionCall(
        function=tool_call.function.name,
        args=arguments,
        id=tool_call.id,
    )


def _openai_to_assistant_message(message: ChatCompletionMessage) -> ChatAssistantMessage:
    if message.tool_calls is not None:
        tool_calls = [_openai_to_tool_call(tool_call) for tool_call in message.tool_calls]
    else:
        tool_calls = None
    return ChatAssistantMessage(role="assistant", content=message.content, tool_calls=tool_calls)


def _function_to_openai(f: Function) -> ChatCompletionToolParam:
    function_definition = FunctionDefinition(
        name=f.name,
        description=f.description,
        parameters=f.parameters.model_json_schema(),
    )
    return ChatCompletionToolParam(type="function", function=function_definition)


@retry(
    wait=wait_random_exponential(multiplier=1, max=40),
    stop=stop_after_attempt(3),
    reraise=True,
    retry=retry_if_not_exception_type(openai.BadRequestError),
)
def chat_completion_request(
    client: openai.OpenAI,
    model: str,
    messages: Sequence[ChatCompletionMessageParam],
    temperature: float | None = 0.0,
):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )

def add_tokens(extra_args, prompt_tokens, completion_tokens):
    extra_args["input_tokens"] += prompt_tokens
    extra_args["output_tokens"] += completion_tokens

class ReActOpenAILLM(BasePipelineElement):
    """LLM pipeline element that uses OpenAI's API for ReAct-style tool use.

    Args:
        client (openai.OpenAI): The OpenAI client.
        model (str): The model name.
        temperature (float | None): Sampling temperature.
    """

    def __init__(self, client: openai.OpenAI, model: str, temperature: float | None = 0.0) -> None:
        super().__init__()
        self.client = client
        self.model = model
        self.temperature = temperature

    def parse_agent_step(self, text: str) -> dict:
        """
        Parse the agent's response into thought, action, and action_input.

        Returns:
            dict: {"thought": str, "action": str, "action_input": Union[dict, str]}
        """
        pattern = r"Thought:\s*(.*?)\s*Action:\s*(.*?)\s*Action Input:\s*(\{.*\})"
        match = re.search(pattern, text, re.DOTALL)

        if not match:
            return {"thought": "", "action": "", "action_input": ""}

        thought = match.group(1).strip()
        action = match.group(2).strip()
        action_input_str = match.group(3).strip()

        try:
            action_input = json.loads(action_input_str)
        except json.JSONDecodeError:
            action_input = action_input_str  # fallback to raw string

        return {
            "thought": thought,
            "action": action,
            "action_input": action_input
        }

    def parse_final_step(self, text: str) -> str:
        """
        Extract the final answer from the LLM response.

        Returns:
            str: The content following 'Final Answer:'
        """
        match = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        """
        Execute a query using the OpenAI chat completion endpoint.

        Returns:
            tuple: Updated query, runtime, env, messages, and extra_args.
        """
        openai_messages = [_message_to_openai(message) for message in messages]
        completion = chat_completion_request(
            client=self.client,
            model=self.model,
            messages=openai_messages,
            temperature=self.temperature,
        )

        reply = completion.choices[0].message.content.strip()

        # Try to parse agent step
        if "Final Answer" in reply:  # fallback to Final Answer if no action detected
            extra_args["final_answer"] = self.parse_final_step(reply)
            extra_args["agent_scratchpad"] += f"Final Answer: {extra_args['final_answer']}\n"
            extra_args["tool_call_name"] = None
            extra_args["tool_call_params"] = None

            output = _openai_to_assistant_message(completion.choices[0].message)
            messages = [*messages, output]
            return query, runtime, env, messages, extra_args

        # Update scratchpad and tool call args
        step = self.parse_agent_step(reply)
        # print(step)
        scratchpad_entry = (
            f"Thought: {step['thought']}\n"
            f"Action: {step['action']}\n"
            f"Action Input: {json.dumps(step['action_input'])}\n"
        )

        extra_args["agent_scratchpad"] = extra_args.get("agent_scratchpad", "") + scratchpad_entry
        extra_args["tool_call_name"] = step["action"]
        extra_args["tool_call_params"] = step["action_input"]

        return query, runtime, env, [], extra_args  