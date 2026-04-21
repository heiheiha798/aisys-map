import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StaticCache,
)


MODEL_PATH = "/data/pretrained_models/Qwen3-0.6B"
PROMPT = (
    "Explain why CUDA Graph is especially useful for small per-step decode work "
    "in LLM inference."
)
DECODE_STEPS = 64
WARMUP_STEPS = 20


def sync_cuda() -> None:
    torch.cuda.synchronize()


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quant_config,
        device_map="cuda",
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model


def prefill_static_cache(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_cache_len: int,
) -> tuple[StaticCache, int]:
    prompt_len = input_ids.shape[1]
    cache = StaticCache(config=model.config, max_cache_len=max_cache_len)
    cache_position = torch.arange(prompt_len, device=input_ids.device, dtype=torch.long)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )

    first_decode_token_id = int(outputs.logits[:, -1, :].argmax(dim=-1).item())
    return cache, first_decode_token_id


def benchmark_cuda_graph(
    model,
    prompt_len: int,
    max_cache_len: int,
    cache: StaticCache,
    first_decode_token_id: int,
    input_dtype: torch.dtype,
    mask_dtype: torch.dtype,
    device: torch.device,
) -> float:
    static_input_ids = torch.zeros((1, 1), device=device, dtype=input_dtype)
    static_attention_mask = torch.zeros((1, max_cache_len), device=device, dtype=mask_dtype)
    static_cache_position = torch.zeros((1,), device=device, dtype=torch.long)
    static_input_ids[0, 0] = first_decode_token_id
    static_attention_mask[:, : prompt_len + 1] = 1
    static_cache_position[0] = prompt_len

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        with torch.no_grad():
            for _ in range(WARMUP_STEPS):
                graph_outputs = model(
                    input_ids=static_input_ids,
                    attention_mask=static_attention_mask,
                    past_key_values=cache,
                    cache_position=static_cache_position,
                    use_cache=True,
                    return_dict=True,
                )
    torch.cuda.current_stream().wait_stream(warmup_stream)
    sync_cuda()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with torch.no_grad():
            graph_outputs = model(
                input_ids=static_input_ids,
                attention_mask=static_attention_mask,
                past_key_values=cache,
                cache_position=static_cache_position,
                use_cache=True,
                return_dict=True,
            )

    sync_cuda()
    token_id = first_decode_token_id
    start = time.perf_counter()
    for step in range(DECODE_STEPS):
        graph.replay()
        token_id = int(graph_outputs.logits[:, -1, :].argmax(dim=-1).item())
        if step + 1 < DECODE_STEPS:
            static_input_ids[0, 0] = token_id
            static_cache_position[0] += 1
            static_attention_mask[:, prompt_len + step + 1] = 1
    sync_cuda()
    end = time.perf_counter()

    return end - start


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment.")

    device = torch.device("cuda")

    try:
        tokenizer, model = load_model_and_tokenizer()
    except Exception as exc:
        raise RuntimeError(
            "Failed to load Qwen3-0.6B with bitsandbytes int4 on HF backend."
        ) from exc

    encoded = tokenizer(PROMPT, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]
    max_cache_len = prompt_len + DECODE_STEPS

    cache, first_decode_token_id = prefill_static_cache(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_cache_len=max_cache_len,
    )

    total_sec = benchmark_cuda_graph(
        model=model,
        prompt_len=prompt_len,
        max_cache_len=max_cache_len,
        cache=cache,
        first_decode_token_id=first_decode_token_id,
        input_dtype=input_ids.dtype,
        mask_dtype=attention_mask.dtype,
        device=device,
    )

    avg_ms = total_sec * 1000.0 / DECODE_STEPS
    tps = DECODE_STEPS / total_sec

    print("experiment: quantization/hf_int4_cuda_graph_decode")
    print(f"model_path: {MODEL_PATH}")
    print("backend: Hugging Face transformers + bitsandbytes int4 + CUDA Graph")
    print(f"prompt token count: {prompt_len}")
    print("decode batch size: 1")
    print(f"warmup steps: {WARMUP_STEPS}")
    print(f"decode steps: {DECODE_STEPS}")
    print("")
    print(f"first decode token id from prefill logits: {first_decode_token_id}")
    print(f"cuda graph avg decode latency: {avg_ms:.3f} ms/token")
    print(f"cuda graph decode throughput: {tps:.3f} tok/s")


if __name__ == "__main__":
    main()
