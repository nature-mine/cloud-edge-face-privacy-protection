"""Load InsightFace's pretrained w600k_r50 ONNX graph as differentiable PyTorch."""

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    def __init__(self, weights, device="cpu"):
        super().__init__()
        import onnx
        from onnx2torch import convert

        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(f"ArcFace weights not found: {weights}")
        graph = onnx.load(str(weights))
        dimensions = graph.graph.input[0].type.tensor_type.shape.dim
        if [dimension.dim_value for dimension in dimensions[1:]] != [3, 112, 112]:
            raise ValueError("Expected InsightFace w600k_r50 with a 3x112x112 RGB input")
        self.backbone = convert(graph).to(device).eval()
        self.requires_grad_(False)
        self.eval()

    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1:] != (3, 112, 112):
            raise ValueError("ArcFace input must have shape (N, 3, 112, 112)")
        return F.normalize(self.backbone(rgb * 2.0 - 1.0), dim=1)


def verify_conversion(model, weights, face):
    """Compare real ONNX inference to the differentiable graph before optimization."""
    import numpy as np
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = min(torch.get_num_threads(), 4)
    session = ort.InferenceSession(
        str(weights), sess_options=options, providers=["CPUExecutionProvider"]
    )
    reference = session.run(
        None, {session.get_inputs()[0].name: (face.detach().cpu().numpy() * 2 - 1)}
    )[0]
    reference /= np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-12)
    with torch.no_grad():
        converted = model(face).cpu().numpy()
    error = float(np.max(np.abs(converted - reference)))
    if error > 1e-4:
        raise RuntimeError(f"ArcFace ONNX conversion failed equivalence check: {error:.6g}")
    return error
