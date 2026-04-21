import argparse
import json
import os
import subprocess


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


def run_llamacpp_batch(
    binary: str,
    model_path: str,
    batch_size: int,
    prompt_len: int,
    decode_tokens: int,
    ctx_size: int,
    gpu: int,
) -> dict:
    cmd = [
        binary,
        "-m",
        model_path,
        "-ngl",
        "999",
        "-fa",
        "on",
        "--backend-sampling",
        "-c",
        str(ctx_size),
        "-b",
        str(max(prompt_len * batch_size, 32)),
        "-ub",
        str(max(prompt_len * batch_size, 32)),
        "-npp",
        str(prompt_len),
        "-ntg",
        str(decode_tokens),
        "-npl",
        str(batch_size),
        "--output-format",
        "jsonl",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["LD_LIBRARY_PATH"] = ":".join(
        [
            "/data/home/tianjianyang/miniconda3/envs/aisys-llamacpp/lib",
            "/data/home/tianjianyang/miniconda3/lib",
            "/usr/local/cuda-12.4/targets/x86_64-linux/lib",
        ]
    )

    proc = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
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

    return {
        "batch_size": batch_size,
        "returncode": proc.returncode,
        "prompt_tps": result.get("speed_pp") if result else None,
        "decode_tps": result.get("speed_tg") if result else None,
        "elapsed_prompt": result.get("t_pp") if result else None,
        "elapsed_decode": result.get("t_tg") if result else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-size sweep for llama.cpp.",
    )
    parser.add_argument(
        "--model",
        default="/data/home/tianjianyang/models/ggufs/Qwen3-0.6B-f16.gguf",
        help="GGUF model path used by llama.cpp",
    )
    parser.add_argument(
        "--binary",
        default=(
            "/data/home/tianjianyang/code/aisys-map/"
            "inference-frameworks/llama.cpp/build-codex/bin/llama-batched-bench"
        ),
        help="llama.cpp binary path",
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
        "--ctx-size",
        type=int,
        default=4096,
        help="Context size for llama.cpp",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8,16",
        help="Comma-separated batch sizes to sweep",
    )
    args = parser.parse_args()
    batch_sizes = parse_batch_sizes(args.batch_sizes)

    results = []
    for batch_size in batch_sizes:
        results.append(
            run_llamacpp_batch(
                binary=args.binary,
                model_path=args.model,
                batch_size=batch_size,
                prompt_len=args.prompt_len,
                decode_tokens=args.decode_tokens,
                ctx_size=args.ctx_size,
                gpu=args.gpu,
            )
        )

    summary = {
        "gpu": args.gpu,
        "model": args.model,
        "prompt_len": args.prompt_len,
        "decode_tokens": args.decode_tokens,
        "ctx_size": args.ctx_size,
        "batch_sizes": batch_sizes,
        "flash_attn": True,
        "batched_bench": True,
        "results": results,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
