from agentdojo.strenum import StrEnum


class ModelsEnum(StrEnum):
    """Currently supported models."""

    GPT_4O_2024_05_13 = "gpt-4o-2024-05-13"
    """GPT-4o"""
    GPT_4O_MINI_2024_07_18 = "gpt-4o-mini-2024-07-18"
    """GPT-4o Mini"""
    GPT_4_0125_PREVIEW = "gpt-4-0125-preview"
    """GPT-4 Turbo Preview 0125"""
    GPT_3_5_TURBO_0125 = "gpt-3.5-turbo-0125"
    """GPT-3.5 Turbo 0125"""
    GPT_4_TURBO_2024_04_09 = "gpt-4-turbo-2024-04-09"
    """GPT-4 Turbo"""
    CLAUDE_3_OPUS_20240229 = "claude-3-opus-20240229"
    """Claude 3 Opus"""
    CLAUDE_3_SONNET_20240229 = "claude-3-sonnet-20240229"
    """Claude 3 Sonnet"""
    CLAUDE_3_5_SONNET_20240620 = "claude-3-5-sonnet-20240620"
    """Claude 3.5 Sonnet"""
    CLAUDE_3_5_SONNET_20241022 = "claude-3-5-sonnet-20241022"
    """Claude 3.5 Sonnet October"""
    CLAUDE_3_7_SONNET_20250219 = "claude-3-7-sonnet-20250219"
    """Claude 3.7 Sonnet"""
    CLAUDE_3_5_HAIKU_20241022 = "claude-3-5-haiku-20241022"
    """Claude 3.5 Haiku October"""
    CLAUDE_3_HAIKU_20240307 = "claude-3-haiku-20240307"
    """Claude 3 Haiku"""
    COMMAND_R_PLUS = "command-r-plus"
    """Command R Plus"""
    COMMAND_R = "command-r"
    """Command R"""
    MISTRALAI_MIXTRAL_8X7B_INSTRUCT_V0_1 = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    """MistralAI Mixtral 8x7B Instruct v0.1"""
    LLAMA_3 = "meta-llama/Llama-3-70b-chat-hf"
    """Llama 3"""
    LLAMA_3_3 = "llama-3.3-70b-instruct"
    """Llama 3.3"""
    LLAMA_3_1 = "meta-llama/Llama-3.1-8B-Instruct"
    """Llama 3.1"""
    LLAMA_2 = "meta-llama/Llama-2-7b-chat-hf"
    """Llama 2"""
    QWEN_2_5_7B = "Qwen/Qwen2.5-7B-Instruct"
    """Qwen 2.5"""
    QWEN_2_5_32B = "Qwen/Qwen2.5-32B-Instruct"
    """Qwen 2.5 32B"""
    QWEN_2_5_72B = "Qwen/Qwen2.5-72B-Instruct"
    """Qwen 2.5 72B"""
    DEEPSEEK_V3 = "deepseek-ai/DeepSeek-V3"
    """DeepSeek V3"""
    DEEPSEEK_R1 = "deepseek-ai/DeepSeek-R1"
    """DeepSeek R1"""
    DEEPSEEK_LLAMA = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    """DeepSeek-R1-Distill-Llama-8B"""
    MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.3"
    """Mistral 7B"""
    PHI_4 = "microsoft/Phi-4-mini-instruct"
    """Phi 4"""
    GEMINI_1_5_PRO_002 = "gemini-1.5-pro-002"
    """Gemini 1.5 Pro 002"""
    GEMINI_1_5_PRO_001 = "gemini-1.5-pro-001"
    """Gemini 1.5 Pro 001"""
    GEMINI_1_5_FLASH_002 = "gemini-1.5-flash-002"
    """Gemini 1.5 Flash 002"""
    GEMINI_1_5_FLASH_001 = "gemini-1.5-flash-001"
    """Gemini 1.5 Flash 001"""
    HERMES_8B = "NousResearch/Hermes-3-Llama-3.1-8B"
    """Hermes 3 Llama-3.1-8B"""
    GRANITE_8B = "ibm-granite/granite-3.2-8b-instruct"
    """IBM Granite 3.2 8B Instruct"""


MODEL_PROVIDERS = {
    ModelsEnum.GPT_4O_2024_05_13: "openai",
    ModelsEnum.GPT_4O_MINI_2024_07_18: "openai",
    ModelsEnum.GPT_3_5_TURBO_0125: "openai",
    ModelsEnum.GPT_4_0125_PREVIEW: "openai",
    ModelsEnum.GPT_4_TURBO_2024_04_09: "openai",
    ModelsEnum.CLAUDE_3_OPUS_20240229: "anthropic",
    ModelsEnum.CLAUDE_3_SONNET_20240229: "anthropic",
    ModelsEnum.CLAUDE_3_5_SONNET_20240620: "anthropic",
    ModelsEnum.CLAUDE_3_5_SONNET_20241022: "anthropic",
    ModelsEnum.CLAUDE_3_5_HAIKU_20241022: "anthropic",
    ModelsEnum.CLAUDE_3_HAIKU_20240307: "anthropic",
    ModelsEnum.CLAUDE_3_7_SONNET_20250219: "anthropic",
    ModelsEnum.COMMAND_R_PLUS: "cohere",
    ModelsEnum.COMMAND_R: "cohere",
    ModelsEnum.MISTRALAI_MIXTRAL_8X7B_INSTRUCT_V0_1: "together",
    ModelsEnum.LLAMA_3: "together-prompting",
    ModelsEnum.LLAMA_3_1: "meta",
    ModelsEnum.LLAMA_2: "meta",
    ModelsEnum.LLAMA_3_3: "meta",
    ModelsEnum.QWEN_2_5_7B: "vllm",
    ModelsEnum.QWEN_2_5_32B: "qdd",
    ModelsEnum.QWEN_2_5_72B: "qdd",
    ModelsEnum.DEEPSEEK_LLAMA: "vllm",
    ModelsEnum.PHI_4: "vllm",
    ModelsEnum.DEEPSEEK_V3: "deepseek",
    ModelsEnum.DEEPSEEK_R1: "deepseek",
    ModelsEnum.MISTRAL_7B: "mistral",
    ModelsEnum.GEMINI_1_5_PRO_001: "google",
    ModelsEnum.GEMINI_1_5_PRO_002: "google",
    ModelsEnum.GEMINI_1_5_FLASH_001: "google",
    ModelsEnum.GEMINI_1_5_FLASH_002: "google",
    ModelsEnum.HERMES_8B: "nousresearch",
    ModelsEnum.GRANITE_8B: "ibm",
}


MODEL_NAMES = {
    "gpt-4o-2024-05-13": "GPT-4",
    "gpt-4o-mini-2024-07-18": "GPT-4",
    "gpt-3.5-turbo-0125": "GPT-3.5",
    "gpt-4-turbo-2024-04-09": "GPT-4",
    "gpt-4-0125-preview": "GPT-4",
    "claude-3-opus-20240229": "Claude",
    "claude-3-sonnet-20240229": "Claude",
    "claude-3-5-sonnet-20240620": "Claude",
    "claude-3-5-sonnet-20241022": "Claude",
    "claude-3-5-haiku-20241022": "Claude",
    "claude-3-haiku-20240307": "Claude",
    "claude-3-7-sonnet-20250219": "Claude",
    "command-r-plus": "Command R+",
    "command-r": "Command R",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "Mixtral",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral",
    "meta-llama/Llama-3-70b-chat-hf": "Llama",
    "llama-3.3-70b-instruct": "Llama",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama",
    "meta-llama/Llama-2-7b-chat-hf": "AI assistant",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen created by Alibaba Cloud.",
    "Qwen/Qwen2.5-32B-Instruct": "Qwen created by Alibaba Cloud.",
    "Qwen/Qwen2.5-72B-Instruct": "Qwen created by Alibaba Cloud.",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "AI assistant",
    "deepseek-ai/DeepSeek-V3": "DeepSeek V3",
    "deepseek-ai/DeepSeek-R1": "DeepSeek R1",
    "microsoft/Phi-4-mini-instruct": "AI assistant",
    "gemini-1.5-pro-002": "AI model developed by Google",
    "gemini-1.5-pro-001": "AI model developed by Google",
    "gemini-1.5-flash-002": "AI model developed by Google",
    "gemini-1.5-flash-001": "AI model developed by Google",
    "NousResearch/Hermes-3-Llama-3.1-8B": "AI assistant",
    "ibm-granite/granite-3.2-8b-instruct": "AI model developed by IBM",
}
