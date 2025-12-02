from transformers import AutoConfig

try:
    config = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
    print(f"Config class: {config.__class__.__name__}")
    print(f"Architectures: {config.architectures}")
    print(f"Max position embeddings: {getattr(config, 'max_position_embeddings', 'Not found')}")
except Exception as e:
    print(f"Error: {e}")
