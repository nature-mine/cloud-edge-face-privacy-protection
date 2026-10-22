"""Face parsing -> frequency PGD -> inverse warp -> saved-image evaluation."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from arcface_model import ArcFace, verify_conversion
from frequency_privacy import AttackConfig, block_dct, protect_face
from model import BiSeNet
from test import CORE_FACE_PART_CLASSES


ARCFACE_LANDMARKS = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041],
], dtype=np.float32)


def tensor_from_rgb(rgb, device):
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().unsqueeze(0).to(device) / 255


def rgb_from_tensor(tensor):
    return (tensor.detach().cpu()[0].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def detect_faces(image_rgb, detector_path):
    height, width = image_rgb.shape[:2]
    scale = min(1.0, 960 / max(height, width))
    resized = cv2.resize(image_rgb, (round(width * scale), round(height * scale)))
    detector = cv2.FaceDetectorYN.create(
        str(detector_path), "", (resized.shape[1], resized.shape[0]), 0.8, 0.3, 5000
    )
    _, detections = detector.detect(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
    if detections is None:
        return []
    faces = []
    for detection in detections:
        points = detection[4:14].reshape(5, 2).copy()
        # YuNet returns image-left eye/mouth corner first, as does ArcFace.
        points[:, 0] *= width / resized.shape[1]
        points[:, 1] *= height / resized.shape[0]
        transform, _ = cv2.estimateAffinePartial2D(points, ARCFACE_LANDMARKS, method=cv2.LMEDS)
        if transform is None or not np.isfinite(transform).all():
            raise RuntimeError("Cannot estimate a valid five-point face alignment")
        faces.append((points, transform.astype(np.float32), float(detection[-1])))
    return faces


def parse_mask(aligned_rgb, parser, device):
    image = tensor_from_rgb(aligned_rgb, device)
    image = F.interpolate(image, size=(512, 512), mode="bilinear", align_corners=False)
    mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    with torch.no_grad():
        parsing = parser((image - mean) / std)[0].argmax(1)[0].cpu().numpy().astype(np.uint8)
    parsing = cv2.resize(parsing, (112, 112), interpolation=cv2.INTER_NEAREST)
    binary = np.isin(parsing, list(CORE_FACE_PART_CLASSES)).astype(np.float32)
    # Feather inward so background and non-target semantic regions stay untouched.
    mask = binary * cv2.GaussianBlur(binary, (5, 5), 1.0)
    return mask, parsing


def inverse_warp_delta(original, aligned_delta, transform):
    """Warp only the masked change; a zero attack must reproduce the original exactly."""
    inverse = cv2.invertAffineTransform(transform)
    return cv2.warpAffine(
        aligned_delta, inverse, (original.shape[1], original.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def quantize_protected(original, delta, epsilon):
    limit = math.floor(epsilon * 255 + 1e-6)
    base = original.astype(np.float32)
    result = np.rint(base + np.clip(delta * 255, -limit, limit))
    return np.clip(result, 0, 255).astype(np.uint8)


def comparison_image(original, protected, path):
    difference = np.abs(protected.astype(np.int16) - original.astype(np.int16))
    amplified = np.clip(difference * 16, 0, 255).astype(np.uint8)
    canvas = Image.new("RGB", (336 * 3, 368), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (array, label) in enumerate([
        (original, "Original"), (protected, "Protected"), (amplified, "Absolute difference x16")
    ]):
        canvas.paste(Image.fromarray(array).resize((336, 336)), (336 * index, 32))
        draw.text((336 * index + 8, 10), label, fill="black")
    canvas.save(path)


def run(args):
    started = time.perf_counter()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    config = AttackConfig(
        steps=args.steps, step_size=args.step_size / 255, epsilon=args.epsilon / 255,
        mid_budget=args.mid_budget, high_budget=args.high_budget,
        bands=args.bands, seed=args.seed,
    )
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    original = np.array(Image.open(args.input).convert("RGB"))
    faces = detect_faces(original, args.detector)
    if not faces:
        raise RuntimeError("No face detected; no protected image was produced")
    print(f"Detected {len(faces)} face(s); device={device}; bands={config.bands}", flush=True)
    parser = BiSeNet(n_classes=19, pretrained=False).to(device).eval()
    parser.load_state_dict(torch.load(args.parsing_weights, map_location=device, weights_only=True))
    parser.requires_grad_(False)
    recognizer = ArcFace(args.arcface_weights, device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    total_delta = np.zeros_like(original, dtype=np.float32)
    report = {
        "input": Path(args.input).name,
        "input_sha256": sha256(args.input),
        "device": str(device),
        "torch_version": torch.__version__,
        "config": asdict(config),
        "detector": "OpenCV YuNet 2023mar",
        "recognizer": "InsightFace buffalo_l w600k_r50 (ArcFace)",
        "arcface_sha256": sha256(args.arcface_weights),
        "detector_sha256": sha256(args.detector),
        "parsing_sha256": sha256(args.parsing_weights),
        "faces": [],
    }
    originals = []
    for index, (landmarks, transform, confidence) in enumerate(faces):
        aligned = cv2.warpAffine(original, transform, (112, 112), flags=cv2.INTER_LINEAR)
        image = tensor_from_rgb(aligned, device)
        if index == 0:
            report["onnx_conversion_max_error"] = verify_conversion(recognizer, args.arcface_weights, image)
        mask_array, parsing = parse_mask(aligned, parser, device)
        if not np.any(mask_array):
            raise RuntimeError(f"No core face parts parsed for face {index}; cannot protect it")
        mask = torch.from_numpy(mask_array).to(device).view(1, 1, 112, 112)
        result = protect_face(image, mask, recognizer, config)
        change = (result.image - image)[0].permute(1, 2, 0).cpu().numpy()
        total_delta += inverse_warp_delta(original, change, transform)
        protected_aligned = rgb_from_tensor(result.image)
        prefix = f"face_{index:02d}"
        Image.fromarray(aligned).save(output / f"{prefix}_original.png")
        Image.fromarray(protected_aligned).save(output / f"{prefix}_protected.png")
        Image.fromarray((mask_array * 255).round().astype(np.uint8)).save(output / f"{prefix}_mask.png")
        Image.fromarray(parsing).save(output / f"{prefix}_parsing.png")
        comparison_image(aligned, protected_aligned, output / f"{prefix}_comparison.png")
        np.savez_compressed(
            output / f"{prefix}_frequency.npz",
            coefficients=block_dct(image).cpu().numpy(),
            delta=result.delta.cpu().numpy(), mask=mask_array, transform=transform,
            block_size=np.array(8), normalization=np.array("orthonormal RGB [0,1]"),
        )
        report["faces"].append({
            "index": index, "detection_confidence": confidence,
            "landmarks": landmarks.tolist(), "transform": transform.tolist(),
            "mask_pixels": int(np.count_nonzero(mask_array)),
            "cosine_before": result.cosine_before,
            "cosine_aligned_float": result.cosine_after,
            "aligned_pixel_linf": float(np.abs(change).max()),
            "history": result.history,
        })
        originals.append((image, transform))
        print(f"Face {index}: aligned cosine {result.cosine_before:.6f} -> {result.cosine_after:.6f}", flush=True)
    protected = quantize_protected(original, total_delta, config.epsilon)
    saved_path = output / f"{Path(args.input).stem}_protected.png"
    Image.fromarray(protected).save(saved_path)
    saved = np.array(Image.open(saved_path).convert("RGB"))
    for face_report, (image, transform) in zip(report["faces"], originals):
        realigned = cv2.warpAffine(saved, transform, (112, 112), flags=cv2.INTER_LINEAR)
        with torch.no_grad():
            similarity = F.cosine_similarity(recognizer(image), recognizer(tensor_from_rgb(realigned, device)))
        face_report["cosine_saved_image"] = float(similarity.item())
        comparison_image(rgb_from_tensor(image), realigned, output / f"face_{face_report['index']:02d}_saved_comparison.png")
        print(f"Face {face_report['index']}: saved-image cosine {similarity.item():.6f}", flush=True)
    difference = saved.astype(np.float32) - original.astype(np.float32)
    mse = float(np.mean(difference ** 2))
    report.update({
        "saved_pixel_linf_255": int(np.abs(difference).max()),
        "full_image_psnr_db": 10 * math.log10(255 ** 2 / mse) if mse else None,
        "seconds": time.perf_counter() - started,
        "output": saved_path.name,
        "limitations": [
            "Single-model similarity is not a calibrated recognition success/failure rate.",
            "Masking, clipping, quantization and resampling can introduce low frequencies in the final image.",
            "AdaFace transfer and dataset-level verification are not evaluated.",
        ],
    })
    with (output / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Saved {saved_path}; pixel Linf={report['saved_pixel_linf_255']}/255; elapsed={report['seconds']:.1f}s", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/example.jpg")
    parser.add_argument("--output", default="res/privacy")
    parser.add_argument("--parsing-weights", default="res/cp/79999_iter.pth")
    parser.add_argument("--arcface-weights", default="weights/w600k_r50.onnx")
    parser.add_argument("--detector", default="weights/face_detection_yunet_2023mar.onnx")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--bands", choices=["high", "mid-high"], default="high")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=8, help="pixel budget in 0..255 units")
    parser.add_argument("--step-size", type=float, default=1, help="DCT update step, divided by 255")
    parser.add_argument("--mid-budget", type=float, default=0.08)
    parser.add_argument("--high-budget", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
