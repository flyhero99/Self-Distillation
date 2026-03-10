"""
vLLM Engine - Connects to vLLM's OpenAI-compatible API for local model inference.

Environment Variables:
    VLLM_API_BASE: Server URL (default: http://localhost:8000/v1)
    VLLM_MODEL_NAME: Model name override (optional, uses the model_name parameter if not set)
    VLLM_ENABLE_THINKING: Set to "true" to enable Qwen3 thinking mode (default: disabled)
"""

import os
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError, InternalServerError
import backoff


@backoff.on_exception(
    backoff.expo,
    (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError),
    max_tries=5
)
def vllm_chat_engine(client, model_name, msg, temperature, top_p, extra_body=None):
    """Send a chat completion request to the vLLM server."""
    response = client.chat.completions.create(
        model=model_name,
        messages=msg,
        temperature=temperature,
        max_tokens=15000,
        top_p=top_p,
        extra_body=extra_body or {},
    )
    return response


class VLLMEngine:
    """Engine for vLLM's OpenAI-compatible API."""

    def __init__(self, model_name):
        """
        Initialize the vLLM engine.

        Args:
            model_name: HuggingFace model name (e.g., "Qwen/Qwen3-32B")
        """
        # Get API base URL from environment, default to localhost
        api_base = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")

        # Allow model name override via environment variable
        self.model_name = os.environ.get("VLLM_MODEL_NAME", model_name)

        # Check if this is a Qwen3 model (for thinking mode handling)
        self.is_qwen3 = "qwen3" in self.model_name.lower()

        # By default, disable thinking for Qwen3 (use /no_think suffix)
        # Set VLLM_ENABLE_THINKING=true to enable thinking mode
        self.enable_thinking = os.environ.get("VLLM_ENABLE_THINKING", "false").lower() == "true"

        # Create OpenAI client pointing to vLLM server
        # vLLM doesn't require an API key, but the OpenAI client expects one
        self.client = OpenAI(
            base_url=api_base,
            api_key="EMPTY",  # vLLM doesn't validate API keys
            max_retries=5,
            timeout=500.0,  # Longer timeout for large models
        )

        print(f"[VLLMEngine] Initialized with model: {self.model_name}")
        print(f"[VLLMEngine] API base: {api_base}")
        if self.is_qwen3:
            print(f"[VLLMEngine] Qwen3 detected - thinking mode: {'enabled' if self.enable_thinking else 'disabled'}")


    def respond(self, user_input, temperature, top_p):
        """
        Generate a response using the vLLM server.

        Args:
            user_input: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            Tuple of (response_text, prompt_tokens, completion_tokens, full_response)
            full_response is the complete model output including thinking (None if thinking is disabled)
        """
        # Prepare messages (no longer need suffix - using enable_thinking parameter)
        messages = user_input

        # Prepare extra parameters for Qwen3 thinking mode
        extra_body = {}
        if self.is_qwen3 and self.enable_thinking:
            extra_body["enable_thinking"] = True
            print("[VLLMEngine] Enabling thinking mode via enable_thinking=true parameter")

        response = vllm_chat_engine(
            self.client,
            self.model_name,
            messages,
            temperature,
            top_p,
            extra_body
        )


        # print("RAW RESPONSE: \n", response)
        message = response.choices[0].message

        # Extract content and reasoning from response
        # With Qwen3 + reasoning parser + enable_thinking:
        # - message.content: Final code/answer
        # - message.reasoning: Thinking process
        content = message.content or ""
        reasoning = getattr(message, 'reasoning', None)

        # Build full response with reasoning followed by content
        full_response = None
        if self.enable_thinking and reasoning:
            # Format: <think>reasoning</think>\n\ncontent
            full_response = f"<think>\n{reasoning}\n</think>\n\n{content}"
            print(f"[VLLMEngine] Thinking traces captured ({len(reasoning)} chars)")
        elif self.enable_thinking:
            # Thinking enabled but no reasoning generated
            print("[VLLMEngine] WARNING: Thinking mode enabled but no reasoning generated")
            full_response = content

        # Log what we extracted
        if reasoning:
            print(f"[VLLMEngine] Reasoning: {len(reasoning)} chars")
        if content:
            print(f"[VLLMEngine] Content: {len(content)} chars")

        return (
            content,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            full_response
        )
