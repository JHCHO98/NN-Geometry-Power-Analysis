import psutil
import os
import onnxruntime as ort
import numpy as np
import time

process=psutil.Process(os.getpid())
process.nice(psutil.HIGH_PRIORITY_CLASS)
process.cpu_affinity([0])

so=ort.SessionOptions()
so.intra_op_num_threads=1
so.inter_op_num_threads=1

session=ort.InferenceSession(
    "model_onnx/mid_balanced.onnx",
    sess_options=so,
    providers=["CPUExecutionProvider"]
    )

WARMUP=200
N=1000
x=np.random.rand(1,3,32,32).astype(np.float32)

for _ in range(WARMUP):
    y=session.run(None, {"input":x})

start=time.perf_counter()

for i in range(N):
    y=session.run(None, {"input":x})

end=time.perf_counter()

elapsed=end-start
avg_latency = elapsed/1000

print(y[0].shape)
print(f"avg_latency: {avg_latency*1000:3f}ms")