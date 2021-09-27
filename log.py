import torch
import time

x = torch.full((1 << 20,), 2.0)

y = x.log()

ITERS = 10
start = time.perf_counter()
for _ in range(ITERS):
    y = x.log()
stop = time.perf_counter()
millis = (stop - start) * 1e3 / ITERS

bytes = x.numel() * x.element_size() * 2
ops = x.numel()
print(f"{millis:.3f} ms {bytes / millis / 1e6:.1f} gb/s {ops / millis / 1e6:.1f} gops/s")

