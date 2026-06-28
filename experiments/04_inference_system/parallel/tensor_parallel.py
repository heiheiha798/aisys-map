import torch


def print_tensor(name: str, tensor: torch.Tensor) -> None:
    print(f"{name} =")
    print(tensor)
    print("")


def column_parallel_demo() -> None:
    print("=== Column Parallel Demo ===")

    # 输入 shape = [batch, in_features]
    x = torch.tensor(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ],
        dtype=torch.int64,
    )

    # 权重 shape = [out_features, in_features]
    w = torch.tensor(
        [
            [1, 0, 2, 0],
            [0, 1, 0, 2],
            [1, 1, 1, 1],
            [2, 0, 0, 1],
        ],
        dtype=torch.int64,
    )
    b = torch.tensor([1, 2, 3, 4], dtype=torch.int64)

    print_tensor("x", x)
    print_tensor("w", w)
    print_tensor("b", b)

    # 全量 linear：y = x @ w^T + b
    full = x @ w.T + b
    print_tensor("full_output", full)

    # column parallel:
    # 沿着输出维切权重，也就是把 out_features 切给不同 rank。
    w_rank0, w_rank1 = w[:2], w[2:]
    b_rank0, b_rank1 = b[:2], b[2:]

    out_rank0 = x @ w_rank0.T + b_rank0
    out_rank1 = x @ w_rank1.T + b_rank1

    print_tensor("w_rank0", w_rank0)
    print_tensor("w_rank1", w_rank1)
    print_tensor("out_rank0", out_rank0)
    print_tensor("out_rank1", out_rank1)

    merged = torch.cat([out_rank0, out_rank1], dim=-1)
    print_tensor("merged_output", merged)

    assert torch.equal(full, merged)
    print("column parallel result matches full linear")
    print("")


def row_parallel_demo() -> None:
    print("=== Row Parallel Demo ===")

    x = torch.tensor(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ],
        dtype=torch.int64,
    )
    w = torch.tensor(
        [
            [1, 0, 2, 0],
            [0, 1, 0, 2],
            [1, 1, 1, 1],
        ],
        dtype=torch.int64,
    )
    b = torch.tensor([1, 2, 3], dtype=torch.int64)

    print_tensor("x", x)
    print_tensor("w", w)
    print_tensor("b", b)

    full = x @ w.T + b
    print_tensor("full_output", full)

    # row parallel:
    # 沿着输入维切权重，也就是把 in_features 切给不同 rank。
    x_rank0, x_rank1 = x[:, :2], x[:, 2:]
    w_rank0, w_rank1 = w[:, :2], w[:, 2:]

    partial_rank0 = x_rank0 @ w_rank0.T
    partial_rank1 = x_rank1 @ w_rank1.T

    print_tensor("x_rank0", x_rank0)
    print_tensor("x_rank1", x_rank1)
    print_tensor("w_rank0", w_rank0)
    print_tensor("w_rank1", w_rank1)
    print_tensor("partial_rank0", partial_rank0)
    print_tensor("partial_rank1", partial_rank1)

    merged = partial_rank0 + partial_rank1 + b
    print_tensor("merged_output", merged)

    assert torch.equal(full, merged)
    print("row parallel result matches full linear")
    print("")


def main() -> None:
    column_parallel_demo()
    row_parallel_demo()


if __name__ == "__main__":
    main()
