import os
import numpy as np
import torch
import onnxruntime as ort

from load_data import get_dataloaders
from FlexibleCNN import FlexibleCNN   # 모델 정의만 들어있는 파일

DEVICE = torch.device("cpu")

BATCH_SIZE = 128

modes = [
    "deep_narrow",
    "shallow_wide",
    "mid_balanced",
    "funnel_wide_to_narrow",
    "uniform",
    "hourglass"
]

_, testloader = get_dataloaders(batch_size=BATCH_SIZE)


def evaluate_pytorch(model, loader):
    model.eval()

    correct = 0
    total = 0

    first_input = None
    first_output = None

    with torch.no_grad():
        for x, y in loader:

            if first_input is None:
                first_input = x.clone()

            output = model(x)

            if first_output is None:
                first_output = output.clone()

            pred = output.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = 100 * correct / total
    return acc, first_input, first_output


def evaluate_onnx(session, loader):

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:

            output = session.run(
                None,
                {"input": x.numpy()}
            )[0]

            pred = np.argmax(output, axis=1)

            correct += np.sum(pred == y.numpy())
            total += y.size(0)

    acc = 100 * correct / total
    return acc


for mode in modes:

    print("=" * 60)
    print(mode)

    ##############################################
    # Load PyTorch model
    ##############################################

    model = FlexibleCNN(mode=mode)

    model.load_state_dict(
        torch.load(
            f"./checkpoints/{mode}_best.pth",
            map_location=DEVICE
        )
    )

    model.eval()

    ##############################################
    # Export ONNX
    ##############################################

    dummy = torch.randn(1, 3, 32, 32)

    onnx_path = f"{mode}.onnx"

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
    )

    ##############################################
    # Accuracy (PyTorch)
    ##############################################

    pytorch_acc, first_input, pytorch_output = evaluate_pytorch(
        model,
        testloader
    )

    ##############################################
    # ONNX Runtime
    ##############################################

    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"]
    )

    ##############################################
    # Accuracy (ONNX)
    ##############################################

    onnx_acc = evaluate_onnx(
        session,
        testloader
    )

    ##############################################
    # Output Difference
    ##############################################

    onnx_output = session.run(
        None,
        {"input": first_input.numpy()}
    )[0]

    pytorch_output = pytorch_output.numpy()

    max_error = np.max(np.abs(pytorch_output - onnx_output))
    mean_error = np.mean(np.abs(pytorch_output - onnx_output))

    ##############################################
    # Result
    ##############################################

    print(f"PyTorch Accuracy : {pytorch_acc:.2f}%")
    print(f"ONNX Accuracy    : {onnx_acc:.2f}%")

    print(f"Accuracy Diff    : {abs(pytorch_acc-onnx_acc):.6f}%")

    print(f"Max Error        : {max_error:.10f}")
    print(f"Mean Error       : {mean_error:.10f}")

    if np.allclose(
        pytorch_output,
        onnx_output,
        atol=1e-5
    ):
        print("✅ Output Verified")
    else:
        print("❌ Output Mismatch")

print("\n모든 모델 검증 완료.")