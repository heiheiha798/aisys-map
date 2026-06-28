import torch


def print_tensor(name: str, tensor: torch.Tensor) -> None:
    print(f"{name} =")
    print(tensor)
    print("")


def expert_forward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return x @ weight.T + bias


def main() -> None:
    # 4 个 token，每个 token 的 hidden dim = 3
    tokens = torch.tensor(
        [
            [1, 0, 2],
            [0, 1, 1],
            [2, 1, 0],
            [1, 2, 1],
        ],
        dtype=torch.int64,
    )

    # gate 给出每个 token 应该去哪个 expert。
    # 这里用最简单的 top-1 routing。
    expert_ids = torch.tensor([0, 1, 0, 1], dtype=torch.int64)

    # expert 0 放在逻辑 rank 0
    w0 = torch.tensor(
        [
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=torch.int64,
    )
    b0 = torch.tensor([1, 2], dtype=torch.int64)

    # expert 1 放在逻辑 rank 1
    w1 = torch.tensor(
        [
            [2, 1, 0],
            [0, 1, 2],
        ],
        dtype=torch.int64,
    )
    b1 = torch.tensor([3, 4], dtype=torch.int64)

    print_tensor("tokens", tokens)
    print_tensor("expert_ids", expert_ids)
    print_tensor("expert0_weight", w0)
    print_tensor("expert1_weight", w1)

    # dispatch:
    # 根据 gate，把 token 分发给不同 expert。
    mask0 = expert_ids == 0
    mask1 = expert_ids == 1
    tokens_e0 = tokens[mask0]
    tokens_e1 = tokens[mask1]

    print_tensor("tokens_for_expert0", tokens_e0)
    print_tensor("tokens_for_expert1", tokens_e1)

    # 每个 expert 各自只处理路由到自己的 token。
    out_e0 = expert_forward(tokens_e0, w0, b0)
    out_e1 = expert_forward(tokens_e1, w1, b1)

    print_tensor("expert0_output", out_e0)
    print_tensor("expert1_output", out_e1)

    # gather:
    # 把 expert 输出按原 token 顺序放回去。
    output = torch.empty((tokens.size(0), 2), dtype=torch.int64)
    output[mask0] = out_e0
    output[mask1] = out_e1

    print_tensor("merged_output", output)

    # 再显式打印 token -> expert 的对应关系。
    for token_idx, expert_id in enumerate(expert_ids.tolist()):
        print(f"token {token_idx} routed to expert {expert_id}")


if __name__ == "__main__":
    main()
