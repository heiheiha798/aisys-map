import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "/data/pretrained_models/Qwen3-0.6B"
PROMPTS = [
    "Explain prefill in one sentence.",
    "Explain the difference between prefill and decode in one short paragraph.",
]


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_forward(model: AutoModelForCausalLM, **kwargs: Any):
    sync_cuda()
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(**kwargs)
    sync_cuda()
    end = time.perf_counter()
    return outputs, end - start


def get_first_layer_kv_shape(past_key_values: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if hasattr(past_key_values, "layers") and len(past_key_values.layers) > 0:
        layer0 = past_key_values.layers[0]
        if hasattr(layer0, "keys") and hasattr(layer0, "values"):
            return tuple(layer0.keys.shape), tuple(layer0.values.shape)

    if hasattr(past_key_values, "to_legacy_cache"):
        legacy = past_key_values.to_legacy_cache()
        if len(legacy) > 0:
            key, value = legacy[0]
            return tuple(key.shape), tuple(value.shape)

    if isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
        key, value = past_key_values[0]
        return tuple(key.shape), tuple(value.shape)

    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)!r}")


def decode_token(tokenizer: AutoTokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False).replace("\n", "\\n")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment.")

    device = torch.device("cuda")

    print("experiment: hf_inference/batching")
    print(f"loading tokenizer from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"loading model from: {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)
    model.eval()

    encoded = tokenizer(PROMPTS, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    lengths = attention_mask.sum(dim=1).tolist()

    print("batch prompts:")
    for idx, prompt in enumerate(PROMPTS):
        print(f"  sample {idx}: len={int(lengths[idx])}, text={prompt}")
    print("")

    print(f"padding_side: {tokenizer.padding_side}")
    print(f"pad_token_id: {tokenizer.pad_token_id}")
    print("")

    print(f"batched input_ids shape: {tuple(input_ids.shape)}")
    print(f"batched attention_mask shape: {tuple(attention_mask.shape)}")
    print("input_ids values:")
    print(input_ids.cpu())
    print("attention_mask values:")
    print(attention_mask.to(torch.int32).cpu())
    print("")

    prefill_outputs, prefill_sec = timed_forward(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )

    past_key_values = prefill_outputs.past_key_values
    k_shape, v_shape = get_first_layer_kv_shape(past_key_values)
    next_tokens = prefill_outputs.logits[:, -1, :].argmax(dim=-1)

    print(f"prefill latency: {prefill_sec * 1000:.3f} ms")
    print(f"prefill logits shape: {tuple(prefill_outputs.logits.shape)}")
    print(f"layer0 K cache shape after batched prefill: {k_shape}")
    print(f"layer0 V cache shape after batched prefill: {v_shape}")
    print("")

    print("first decode token per sample:")
    for idx, token_id in enumerate(next_tokens.tolist()):
        print(
            f"  sample {idx}: token_id={token_id}, token_text={decode_token(tokenizer, token_id)!r}"
        )
    print("")

    next_input_ids = next_tokens.unsqueeze(1)
    next_attention_mask = torch.cat(
        [
            attention_mask,
            torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype),
        ],
        dim=1,
    )

    decode_outputs, decode_sec = timed_forward(
        model,
        input_ids=next_input_ids,
        attention_mask=next_attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
    )

    next_k_shape, next_v_shape = get_first_layer_kv_shape(decode_outputs.past_key_values)
    second_tokens = decode_outputs.logits[:, -1, :].argmax(dim=-1)

    print("one decode step after batched prefill:")
    print(f"  decode input_ids shape: {tuple(next_input_ids.shape)}")
    print(f"  decode attention_mask shape: {tuple(next_attention_mask.shape)}")
    print(f"  decode latency: {decode_sec * 1000:.3f} ms")
    print(f"  layer0 K cache shape after decode: {next_k_shape}")
    print(f"  layer0 V cache shape after decode: {next_v_shape}")
    print("")

    print("second decode token per sample:")
    for idx, token_id in enumerate(second_tokens.tolist()):
        print(
            f"  sample {idx}: token_id={token_id}, token_text={decode_token(tokenizer, token_id)!r}"
        )


if __name__ == "__main__":
    main()
