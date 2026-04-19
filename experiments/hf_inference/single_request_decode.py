import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "/data/pretrained_models/Qwen3-0.6B"
PROMPT = (
    "Explain the difference between prefill and decode in LLM inference "
    "in two short sentences."
)
MAX_NEW_TOKENS = 12


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

    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        key = past_key_values.key_cache[0]
        value = past_key_values.value_cache[0]
        return tuple(key.shape), tuple(value.shape)

    if hasattr(past_key_values, "to_legacy_cache"):
        legacy_cache = past_key_values.to_legacy_cache()
        if len(legacy_cache) > 0:
            key, value = legacy_cache[0]
            return tuple(key.shape), tuple(value.shape)

    if isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
        key, value = past_key_values[0]
        return tuple(key.shape), tuple(value.shape)

    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)!r}")


def decode_token(tokenizer: AutoTokenizer, token_id: int) -> str:
    text = tokenizer.decode([token_id], skip_special_tokens=False)
    return text.replace("\n", "\\n")


def print_model_summary(model: AutoModelForCausalLM) -> None:
    config = model.config

    print("model summary:")
    print(f"  hidden_size={config.hidden_size}")
    print(f"  num_attention_heads={config.num_attention_heads}")
    print(f"  num_key_value_heads={config.num_key_value_heads}")
    print(f"  head_dim={config.head_dim}")
    print(f"  use_cache={config.use_cache}")
    print("")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment.")

    device = torch.device("cuda")

    print("experiment: hf_inference/single_request_decode")
    print(f"loading tokenizer from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    print(f"loading model from: {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)
    model.eval()

    encoded = tokenizer(PROMPT, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    print_model_summary(model)
    print(f"prompt: {PROMPT}")
    print(f"prompt token count: {input_ids.shape[1]}")
    print(f"prefill input_ids shape: {tuple(input_ids.shape)}")
    print(f"prefill attention_mask shape: {tuple(attention_mask.shape)}")

    prefill_outputs, prefill_sec = timed_forward(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )

    past_key_values = prefill_outputs.past_key_values
    first_k_shape, first_v_shape = get_first_layer_kv_shape(past_key_values)
    first_next_token = int(prefill_outputs.logits[:, -1, :].argmax(dim=-1).item())

    print(f"prefill latency: {prefill_sec * 1000:.3f} ms")
    print(f"prefill logits shape: {tuple(prefill_outputs.logits.shape)}")
    print(f"layer0 K cache shape after prefill: {first_k_shape}")
    print(f"layer0 V cache shape after prefill: {first_v_shape}")
    print(
        "first decode token from prefill logits: "
        f"id={first_next_token}, text={decode_token(tokenizer, first_next_token)!r}"
    )

    generated_ids = [first_next_token]
    next_token = torch.tensor([[first_next_token]], device=device, dtype=input_ids.dtype)
    running_attention_mask = torch.cat(
        [
            attention_mask,
            torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype),
        ],
        dim=1,
    )

    print("")
    print("decode steps:")

    for step in range(1, MAX_NEW_TOKENS):
        outputs, decode_sec = timed_forward(
            model,
            input_ids=next_token,
            attention_mask=running_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

        past_key_values = outputs.past_key_values
        current_k_shape, current_v_shape = get_first_layer_kv_shape(past_key_values)
        next_token_id = int(outputs.logits[:, -1, :].argmax(dim=-1).item())

        print(
            f"step={step:02d} "
            f"decode_input_shape={tuple(next_token.shape)} "
            f"decode_attention_mask_shape={tuple(running_attention_mask.shape)} "
            f"latency={decode_sec * 1000:.3f} ms "
            f"produced_token_id={next_token_id} "
            f"text={decode_token(tokenizer, next_token_id)!r} "
            f"layer0_k_shape={current_k_shape} "
            f"layer0_v_shape={current_v_shape}"
        )

        generated_ids.append(next_token_id)
        next_token = torch.tensor([[next_token_id]], device=device, dtype=input_ids.dtype)
        running_attention_mask = torch.cat(
            [
                running_attention_mask,
                torch.ones((running_attention_mask.shape[0], 1), device=device, dtype=running_attention_mask.dtype),
            ],
            dim=1,
        )

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("")
    print("final generated continuation:")
    print(generated_text)


if __name__ == "__main__":
    main()
