import sys
import os
sys.path.append(os.getcwd())

from morphkv.monkeypatch import patch_morphkv
import transformers.models.llama.modeling_llama as llama_modeling
from morphkv.models.patch_smolm import SmolLMAttentionMorph

print("Before patch:")
print(f"LlamaAttention: {llama_modeling.LlamaAttention}")

print("Applying patch...")
patch_morphkv()

print("After patch:")
print(f"LlamaAttention: {llama_modeling.LlamaAttention}")

if llama_modeling.LlamaAttention == SmolLMAttentionMorph:
    print("SUCCESS: LlamaAttention patched with SmolLMAttentionMorph")
else:
    print("FAILURE: LlamaAttention NOT patched correctly")
