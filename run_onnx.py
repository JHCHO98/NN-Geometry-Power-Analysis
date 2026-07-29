import psutil
import os
import onnxruntime as ort
import numpy as np

process=psutil.Process(os.getpid())
process.nice(psutil.HIGH_PRIORITY_CLASS)

session=ort.InferenceSession("mid_balanced.onnx")

x=np.random.rand(1,3,32,32).astype(np.float32)

for i in range(100000):
    y=session.run(None, {"input":x})

print(y[0].shape)