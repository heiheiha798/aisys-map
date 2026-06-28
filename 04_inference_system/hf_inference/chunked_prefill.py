import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "/data/pretrained_models/Qwen3-0.6B"
PROMPT = (
    "Chunked prefill feeds a long prompt into the model in several smaller chunks "
    "instead of one full prefill pass."
)
CHUNK_SIZE = 8


def main() -> None:
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)
    model.eval()

    encoded = tokenizer(PROMPT, return_tensors="pt")
    full_input_ids = encoded["input_ids"].to(device)
    full_attention_mask = encoded["attention_mask"].to(device)
    prompt_len = full_input_ids.shape[1]

    past_key_values = None
    chunked_outputs = None
    for chunk_start in range(0, prompt_len, CHUNK_SIZE):
        chunk_end = min(prompt_len, chunk_start + CHUNK_SIZE)
        chunk_input_ids = full_input_ids[:, chunk_start:chunk_end]
        chunk_attention_mask = full_attention_mask[:, :chunk_end]

        with torch.no_grad():
            chunked_outputs = model(
                input_ids=chunk_input_ids,
                attention_mask=chunk_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
        past_key_values = chunked_outputs.past_key_values

    if chunked_outputs is None:
        raise RuntimeError("chunked prefill produced no outputs.")

    _ = int(chunked_outputs.logits[:, -1, :].argmax(dim=-1).item())


if __name__ == "__main__":
    main()
