import numpy as np
import tvm
from tvm import te, auto_scheduler

manual = False
N = 1 << 28

@auto_scheduler.register_workload
def kernel(N):
    A = te.placeholder((N,), name="A")
    B = te.compute((N,), lambda n: A[n] * te.tanh(te.log(te.exp(A[n]))), name="B")
    return [A, B]

if manual:
    A, B = kernel(N)
    s = te.create_schedule(B.op)
    #o, i = s[B].split(s[B].op.axis[0], 1024)
    o, i = s[B].split(s[B].op.axis[0], 4 *1024)
    i, v = s[B].split(i, 4)
    s[B].bind(o, te.thread_axis("blockIdx.x"))
    s[B].bind(i, te.thread_axis("threadIdx.x"))
    s[B].vectorize(v)
    print(tvm.lower(s, [A, B], simple_mode=True))

    f = tvm.build(s, [A, B], "cuda")
    print(f.imported_modules[0].get_source())
    #dev = tvm.cuda(0)
    dev = tvm.cuda(0)
else:
    target = tvm.target.Target("cuda")
    task = auto_scheduler.SearchTask(func=kernel, args=(N,), target=target)
    log_file = "mish.json"
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=100)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=10,  # change this to 1000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    task.tune(tune_option)
    sch, args = task.apply_best(log_file)
    del measure_ctx
    print(tvm.lower(sch, args, simple_mode=True))
    f = tvm.build(sch, args, target)
    print(f.imported_modules[0].get_source())
    
dev = tvm.cuda(0)
a = tvm.nd.array(np.random.uniform(size=(N,)).astype(np.float32), dev)
b = tvm.nd.array(np.zeros((N,), dtype=np.float32), dev)
f(a, b)
evaluator = f.time_evaluator(f.entry_name, dev, number=100)
ms = evaluator(a, b).mean * 1e3
gbps = (N * 2 * 4) / ms / 1e6
print(f"kernel: {ms} ms, {gbps} gb/s")
