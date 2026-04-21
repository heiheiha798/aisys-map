import atexit
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NANOVLLM_ROOT = ROOT / "experiments" / "nano-vllm"

QWEN3_06B_HELLO_10 = [
    14990,
    23811,
    23811,
    23811,
    23811,
    23811,
    23811,
    23811,
    23811,
    23811,
]


def parse_batch_sizes(raw: str) -> list[int]:
    batch_sizes = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"batch size must be positive, got {value}")
        batch_sizes.append(value)
    if not batch_sizes:
        raise ValueError("at least one batch size is required")
    return batch_sizes


def build_prompt(prompt_len: int) -> str:
    return " ".join(["hello"] * prompt_len)


def build_prompt_ids(model_path: str, prompt_len: int) -> list[int]:
    from transformers import AutoTokenizer

    if model_path.rstrip("/").endswith("Qwen3-0.6B") and prompt_len == 10:
        return QWEN3_06B_HELLO_10.copy()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    prompt = build_prompt(prompt_len)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) != prompt_len:
        raise ValueError(
            f"expected {prompt_len} prompt tokens, got {len(prompt_ids)} "
            f"for prompt={prompt!r}"
        )
    return prompt_ids


def run_nanovllm_batch(
    model_path: str,
    batch_size: int,
    prompt_token_ids: list[int],
    decode_tokens: int,
    temperature: float,
    warmup_tokens: int,
    seed: int,
    enforce_eager: bool,
    gpu_memory_utilization: float,
) -> dict:
    import torch

    if str(NANOVLLM_ROOT) not in sys.path:
        sys.path.insert(0, str(NANOVLLM_ROOT))

    import engine.model_runner as model_runner_mod
    from engine.llm_engine import LLM
    from engine.sequence import SamplingParams
    from engine.sequence import Sequence

    if batch_size <= 8:
        engine_batch_size = batch_size
    else:
        engine_batch_size = ((batch_size + 15) // 16) * 16

    orig_warmup_model = model_runner_mod.ModelRunner.warmup_model

    def patched_warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens = self.config.max_num_batched_tokens
        max_model_len = self.config.max_model_len
        seq_len = min(len(prompt_token_ids), max_num_batched_tokens, max_model_len)
        num_seqs = min(batch_size, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    model_runner_mod.ModelRunner.warmup_model = patched_warmup_model

    torch.manual_seed(seed)
    try:
        llm = LLM(
            model_path,
            enforce_eager=enforce_eager,
            max_num_seqs=engine_batch_size,
            max_num_batched_tokens=max(len(prompt_token_ids) * engine_batch_size, 256),
            max_model_len=max(len(prompt_token_ids) + decode_tokens + 16, 256),
            gpu_memory_utilization=gpu_memory_utilization,
        )
        warmup_params = SamplingParams(
            temperature=temperature,
            max_tokens=warmup_tokens,
        )
        warmup_prompts = [prompt_token_ids for _ in range(batch_size)]
        warmup_sampling = [warmup_params for _ in range(batch_size)]
        llm.generate(warmup_prompts, warmup_sampling, use_tqdm=False)

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=decode_tokens,
        )
        tracked_seq_ids = set()
        tracked_seqs = []
        for _ in range(batch_size):
            llm.add_request(prompt_token_ids, sampling_params)
            seq = llm.scheduler.waiting[-1]
            tracked_seq_ids.add(seq.seq_id)
            tracked_seqs.append(seq)

        outputs, num_tokens = llm.step()
        if num_tokens <= 0:
            raise RuntimeError("nano-vllm first step should be prefill.")

        if outputs:
            for seq_id, _ in outputs:
                tracked_seq_ids.discard(seq_id)

        torch.cuda.synchronize()
        start = time.perf_counter()
        while tracked_seq_ids:
            outputs, _ = llm.step()
            for seq_id, _ in outputs:
                tracked_seq_ids.discard(seq_id)
        torch.cuda.synchronize()
        end = time.perf_counter()
    finally:
        model_runner_mod.ModelRunner.warmup_model = orig_warmup_model
        if "llm" in locals():
            atexit.unregister(llm.exit)
            llm.exit()

    total_decode_tokens = sum(max(seq.num_completion_tokens - 1, 0) for seq in tracked_seqs)
    elapsed = end - start
    return {
        "batch_size": batch_size,
        "elapsed": elapsed,
        "total_decode_tokens": total_decode_tokens,
        "decode_tps": total_decode_tokens / elapsed if elapsed > 0 else None,
    }


def run_nanovllm_batch_subprocess(
    script_path: Path,
    model_path: str,
    batch_size: int,
    prompt_len: int,
    decode_tokens: int,
    temperature: float,
    warmup_tokens: int,
    seed: int,
    enforce_eager: bool,
    gpu_memory_utilization: float,
    gpu: int,
) -> dict:
    cmd = [
        sys.executable,
        str(script_path),
        "--internal-single",
        "--model",
        model_path,
        "--gpu",
        str(gpu),
        "--prompt-len",
        str(prompt_len),
        "--decode-tokens",
        str(decode_tokens),
        "--temperature",
        str(temperature),
        "--warmup-tokens",
        str(warmup_tokens),
        "--seed",
        str(seed),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--internal-batch-size",
        str(batch_size),
    ]
    if enforce_eager:
        cmd.append("--enforce-eager")

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"nano-vllm batch subprocess failed for batch_size={batch_size}:\n{proc.stdout}"
        )

    result = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue

    if result is None:
        raise RuntimeError(
            f"nano-vllm batch subprocess produced no JSON for batch_size={batch_size}:\n{proc.stdout}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-size sweep for nano-vllm.",
    )
    parser.add_argument(
        "--model",
        default="/data/pretrained_models/Qwen3-0.6B",
        help="HF model path used by nano-vllm",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=7,
        help="Physical GPU id",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=10,
        help="Prompt length in tokens",
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=100,
        help="Number of generated tokens per sequence",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8,16",
        help="Comma-separated batch sizes to sweep",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for nano-vllm warmup and generation",
    )
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=5,
        help="Warmup decode tokens for nano-vllm",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Torch RNG seed for nano-vllm",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization passed to nano-vllm",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable nano-vllm graph path",
    )
    parser.add_argument(
        "--internal-single",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--internal-batch-size",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    batch_sizes = parse_batch_sizes(args.batch_sizes)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    prompt_token_ids = build_prompt_ids(args.model, args.prompt_len)

    if args.internal_single:
        if args.internal_batch_size is None:
            raise ValueError("--internal-batch-size is required with --internal-single")
        result = run_nanovllm_batch(
            model_path=args.model,
            batch_size=args.internal_batch_size,
            prompt_token_ids=prompt_token_ids,
            decode_tokens=args.decode_tokens,
            temperature=args.temperature,
            warmup_tokens=args.warmup_tokens,
            seed=args.seed,
            enforce_eager=args.enforce_eager,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        print(json.dumps(result))
        return

    results = []
    script_path = Path(__file__).resolve()
    for batch_size in batch_sizes:
        results.append(
            run_nanovllm_batch_subprocess(
                script_path=script_path,
                model_path=args.model,
                batch_size=batch_size,
                prompt_len=args.prompt_len,
                decode_tokens=args.decode_tokens,
                temperature=args.temperature,
                warmup_tokens=args.warmup_tokens,
                seed=args.seed,
                enforce_eager=args.enforce_eager,
                gpu_memory_utilization=args.gpu_memory_utilization,
                gpu=args.gpu,
            )
        )

    summary = {
        "gpu": args.gpu,
        "model": args.model,
        "prompt_len": args.prompt_len,
        "decode_tokens": args.decode_tokens,
        "batch_sizes": batch_sizes,
        "graph": not args.enforce_eager,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "results": results,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
