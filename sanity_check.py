import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
from torchmetrics.text.perplexity import Perplexity
from tqdm import tqdm

def evaluate_perplexity(model, tokenizer, dataset, name, device, max_length=1024, stride=512, batch_size=8):
    """
    Evaluates perplexity on a dataset using TorchMetrics.
    """
    print(f"\n--- Evaluating: {name} ---")
    
    # Encode the entire dataset into a single long tensor of token IDs
    # (Standard approach for perplexity to handle boundaries correctly)
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    
    seq_len = input_ids.size(1)
    metric = Perplexity(ignore_index=-100).to(device)
    
    # Sliding window approach
    # We define a loop that moves a window of size `max_length` across the text
    # The stride is smaller than max_length to provide context for the prediction
    
    total_steps = (seq_len - 1) // stride
    
    # We process in chunks, but since we need to accumulate metric state, 
    # we can just run the forward pass and update the metric.
    # Note: TorchMetrics Perplexity expects (preds, target).
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, seq_len, stride), desc="Processing chunks"):
            # Calculate start and end indices
            begin_loc = i
            end_loc = min(begin_loc + max_length, seq_len)
            
            # If the chunk is too small (e.g., end of dataset), skip or handle
            if end_loc - begin_loc < 2:
                break
                
            trg_len = end_loc - i  # How many tokens we are actually predicting in this window
            
            # Prepare input tensor
            input_chunk = input_ids[:, begin_loc:end_loc]
            
            # Forward pass
            outputs = model(input_chunk)
            logits = outputs.logits
            
            # Shift logits and labels for causal language modeling
            # Logits: predict next token, so we take [:, :-1, :]
            # Targets: are the next token, so we take [:, 1:]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_chunk[..., 1:].contiguous()
            
            # Update metric
            # Torchmetrics expects (preds, target)
            # preds shape: (N, L, C), target shape: (N, L)
            metric.update(shift_logits, shift_labels)
            
            if end_loc == seq_len:
                break

    # Compute final perplexity
    ppl = metric.compute()
    print(f"Dataset: {name} | Perplexity: {ppl.item():.2f}")
    return ppl.item()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Model and Tokenizer
    model_id = "openai-community/gpt2"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)

    # 2. Load Datasets
    # WikiText-2 (Test split)
    print("Loading WikiText-2...")
    wt2 = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # WikiText-103 (Test split) - Note: This is large
    print("Loading WikiText-103...")
    wt103 = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

    # TinyStories (Validation split)
    print("Loading TinyStories...")
    # Using validation split as it is smaller and sufficient for eval
    ts = load_dataset("roneneldan/TinyStories", split="validation")

    # 3. Evaluate
    results = {}
    results["WikiText-2"] = evaluate_perplexity(model, tokenizer, wt2, "WikiText-2", device)
    results["WikiText-103"] = evaluate_perplexity(model, tokenizer, wt103, "WikiText-103", device)
    results["TinyStories"] = evaluate_perplexity(model, tokenizer, ts, "TinyStories", device)

    print("\n=== Final Results ===")
    for k, v in results.items():
        print(f"{k}: {v:.2f}")

if __name__ == "__main__":
    main()