import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "plot" / "data" / "decode_batch_sweep.json"
PDF_PATH = ROOT / "plot" / "decode_batch_sweep.pdf"


def load_series(items):
    xs = [item["batch_size"] for item in items]
    ys = [item["decode_tps"] for item in items]
    return xs, ys


def main() -> None:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    nano_x, nano_y = load_series(payload["results"]["nano_vllm"])
    llama_x, llama_y = load_series(payload["results"]["llama_cpp"])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.6, 5.4), constrained_layout=True)

    ax.plot(
        nano_x,
        nano_y,
        marker="o",
        linewidth=2.4,
        markersize=7,
        color="#0F766E",
        label="nano-vllm",
    )
    ax.plot(
        llama_x,
        llama_y,
        marker="s",
        linewidth=2.4,
        markersize=6.5,
        color="#B45309",
        label="llama.cpp",
    )

    for x, y in zip(nano_x, nano_y):
        ax.annotate(f"{y:.0f}", (x, y), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=9, color="#0F766E")
    for x, y in zip(llama_x, llama_y):
        ax.annotate(f"{y:.0f}", (x, y), xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=9, color="#B45309")

    ax.set_title("Decode Throughput Sweep on GPU7", fontsize=15, pad=12)
    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Decode Throughput (tok/s)", fontsize=12)
    ax.set_xticks(payload["batch_sizes"])
    ax.legend(frameon=True)

    fig.savefig(PDF_PATH, format="pdf", bbox_inches="tight")
    print(PDF_PATH)


if __name__ == "__main__":
    main()
