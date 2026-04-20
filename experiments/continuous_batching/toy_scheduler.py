from collections import deque
from dataclasses import dataclass


@dataclass
class Request:
    req_id: str
    prompt_len: int
    decode_len: int
    arrival_step: int
    prefetched: int = 0
    decoded: int = 0

    @property
    def prefill_done(self) -> bool:
        return self.prefetched >= self.prompt_len

    @property
    def finished(self) -> bool:
        return self.prefill_done and self.decoded >= self.decode_len


class ToyContinuousBatchScheduler:
    def __init__(self, max_batch_tokens: int, max_batch_reqs: int):
        self.max_batch_tokens = max_batch_tokens
        self.max_batch_reqs = max_batch_reqs
        self.waiting: deque[Request] = deque()
        self.running: deque[Request] = deque()
        self.time_step = 0

    def add_arrivals(self, arrivals: list[Request]) -> None:
        for req in arrivals:
            if req.arrival_step == self.time_step:
                self.waiting.append(req)
                print(f"[step {self.time_step}] arrival: {req.req_id}")

    def step(self) -> None:
        print(f"\n=== step {self.time_step} ===")
        print("waiting before:", [r.req_id for r in self.waiting])
        print("running before:", [r.req_id for r in self.running])

        scheduled = []
        mode = None
        remaining_tokens = self.max_batch_tokens

        # 最小 toy 版：
        # 先尝试塞 prefill；如果没塞进去任何 prefill，再做 decode。
        while self.waiting and len(scheduled) < self.max_batch_reqs:
            req = self.waiting[0]
            need = req.prompt_len - req.prefetched
            if need <= 0:
                self.waiting.popleft()
                self.running.append(req)
                continue
            if remaining_tokens == 0:
                break
            if need > remaining_tokens and scheduled:
                break

            chunk = min(need, remaining_tokens)
            scheduled.append((req, "prefill", chunk))
            mode = "prefill"
            remaining_tokens -= chunk
            req.prefetched += chunk

            if req.prefill_done:
                self.waiting.popleft()
                self.running.append(req)
            else:
                break

        if not scheduled:
            decode_round_size = min(len(self.running), self.max_batch_reqs)
            for _ in range(decode_round_size):
                req = self.running.popleft()
                if req.finished:
                    continue
                scheduled.append((req, "decode", 1))
                mode = "decode"
                req.decoded += 1
                if not req.finished:
                    self.running.append(req)

        if not scheduled:
            print("scheduled: []")
        else:
            print("mode:", mode)
            print(
                "scheduled:",
                [(req.req_id, action, amount) for req, action, amount in scheduled],
            )

        print(
            "state after:",
            {
                req.req_id: {
                    "prefetched": req.prefetched,
                    "decoded": req.decoded,
                    "prefill_done": req.prefill_done,
                    "finished": req.finished,
                }
                for req in list(self.waiting) + list(self.running)
            },
        )
        print("waiting after:", [r.req_id for r in self.waiting])
        print("running after:", [r.req_id for r in self.running])

        self.time_step += 1


def main() -> None:
    arrivals = [
        Request(req_id="A", prompt_len=6, decode_len=4, arrival_step=0),
        Request(req_id="B", prompt_len=4, decode_len=3, arrival_step=1),
        Request(req_id="C", prompt_len=5, decode_len=2, arrival_step=3),
    ]

    scheduler = ToyContinuousBatchScheduler(max_batch_tokens=6, max_batch_reqs=2)

    for _ in range(10):
        scheduler.add_arrivals(arrivals)
        scheduler.step()


if __name__ == "__main__":
    main()
