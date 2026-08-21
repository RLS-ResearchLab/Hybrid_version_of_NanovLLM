from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    def __init__(self, model, **kwargs):
        # Forward all Config fields through LLMEngine; includes the MoE
        # quantization controls (use_moe_w8a8, moe_w8a8_weight_group_size)
        # when provided.
        super().__init__(model, **kwargs)
