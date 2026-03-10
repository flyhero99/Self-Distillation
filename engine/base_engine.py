class LLMEngine():
    def __init__(self, llm_engine_name):
        self.llm_engine_name = llm_engine_name
        self.engine = None
        if llm_engine_name.startswith("gpt") or llm_engine_name.startswith("o1"):
            from engine.openai_engine import OpenaiEngine
            self.engine = OpenaiEngine(llm_engine_name)
        elif llm_engine_name.startswith("vllm:"):
            from engine.vllm_engine import VLLMEngine
            model_name = llm_engine_name.replace("vllm:", "", 1)
            self.engine = VLLMEngine(model_name)
        else:
            from engine.bedrock_engine import BedrockEngine
            self.engine = BedrockEngine(llm_engine_name)

    def respond(self, user_input, temperature, top_p):
        result = self.engine.respond(user_input, temperature, top_p)
        # Normalize to 4-tuple: (content, prompt_tokens, completion_tokens, full_response)
        # Non-thinking engines return 3-tuple; pad with None for full_response
        if len(result) == 3:
            return (*result, None)
        return result