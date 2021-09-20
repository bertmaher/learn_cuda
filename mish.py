import torch
import torch.nn.functional as F

N = 1 << 28
x = torch.full((N,), 3.0).cuda()
y = torch.empty_like(x)

start = torch.cuda.Event(enable_timing=True)
stop = torch.cuda.Event(enable_timing=True)

durs = []
for _ in range(500):
    start.record()
    y = F.mish(x)
    stop.record()
    stop.synchronize()
    durs.append(start.elapsed_time(stop))

for dur in durs[:20]:
    print(dur)
dur = torch.sum(torch.tensor(durs)).item() / 500 
print(f"duration (ms): {dur}")
print(f"gb/s: {N * 4 * 3 / dur / 1e6}")
