# Frequency-domain face privacy experiment

This adds a runnable continuation of the existing face parsing and 8x8 DCT demo.
It is an experiment, not evidence that recognition is reliably defeated.

## Run

Use Python >= 3.11 and matching PyTorch/torchvision builds. CPU is supported.
Install in a virtual environment; on Windows, keep its absolute path short
because ONNX includes deeply nested test assets.

```sh
python -m pip install -r requirements-privacy.txt
python download_privacy_models.py
python privacy_demo.py --bands high --output res/privacy/high
python privacy_demo.py --bands mid-high --output res/privacy/mid_high
python -m pytest tests -q
```

The downloader verifies SHA256 and reuses verified files. Standard `HTTPS_PROXY`
configuration can be used when downloads require a proxy. No model weights are
uploaded to Git. The original BiSeNet checkpoint is already in the repository.
Loading that full checkpoint skips the redundant ResNet18 pretrained download;
training constructors retain their previous pretrained default.

For another image, pass `--input path/to/image.jpg`. No detected face or an empty
core-parts mask produces an error instead of silently claiming protection.
Multiple detected faces are processed, with a global pixel budget applied after
combining their changes. CPU thread count defaults to 4.

## Pipeline and implementation choices

1. OpenCV YuNet detects faces and five landmarks. This is a lightweight
   replacement for the document's suggested RetinaFace, not a RetinaFace
   implementation. A similarity transform aligns each face to the InsightFace
   112x112 ArcFace template.
2. The repository's BiSeNet checkpoint segments the aligned face at 512x512.
   Classes `{2, 3, 4, 5, 10, 11, 12, 13}` define the core facial parts. The mask
   is resized to 112x112 and feathered inward, leaving non-target pixels zero.
3. RGB floats in [0, 1] undergo orthonormal 8x8 DCT using differentiable PyTorch
   matrix operations. The existing OpenCV grayscale NPZ remains a separate
   visualization artifact: its centered 0..255 coefficients cannot be used
   directly with the new RGB budget units.
4. Frequency positions are split by `u + v`. DC=0 and low=1..2 are frozen.
   Mid=3..6 is available only with `--bands mid-high`; high>=7 is always enabled.
5. ArcFace's official `buffalo_l/w600k_r50.onnx` is converted by `onnx2torch`.
   Its parameters and normalization statistics are frozen, while gradients
   through its input are retained. RGB is normalized to [-1, 1], without a
   channel swap. Before optimization, the converted graph is compared against
   ONNX Runtime on the real aligned image; normalized embedding error must
   be <= 1e-4.
6. The loss is cosine similarity between normalized clean and candidate
   embeddings. A fixed-seed, tiny random coefficient initialization avoids
   the zero-gradient stationary point at identical embeddings. This is an
   intentional deviation from the document's all-zero initialization.
7. Each PGD update subtracts the signed gradient, clips coefficients to their
   band budgets, then uniformly scales coefficients when the masked IDCT change
   would exceed the pixel budget. Uniform scaling preserves coefficient support.
   The best-scoring feasible iterate is retained, including the clean baseline.
8. The final aligned image is `clip(x + mask * IDCT(delta), 0, 1)`. Only this
   image's change from the clean aligned face is inverse-warped and added to the
   original. The mask is already applied and is not multiplied a second time.
   This avoids changing clean pixels through a warp/unwarp round trip.
9. The combined full-image change is bounded, rounded to uint8 and saved as PNG.
   The PNG is reopened, aligned with the original fixed transform and evaluated
   again. Thus `cosine_saved_image` includes interpolation and quantization
   effects, while `cosine_aligned_float` measures the optimization target.

The final image need not have zero low-frequency *pixel-change* coefficients:
spatial masking, clipping, inverse warping and quantization mix frequencies.
The exact zero-frequency-support guarantee applies to `delta` before those
operations. This distinction is not an implementation success metric.

## Parameters

| Argument | Default | Units / effect |
| --- | --- | --- |
| `--bands` | `high` | `high` or `mid-high` |
| `--steps` | 20 | PGD updates |
| `--epsilon` | 8 | Pixel budget, divided by 255 |
| `--step-size` | 1 | Coefficient step, divided by 255 |
| `--mid-budget` | 0.08 | Orthonormal RGB DCT units |
| `--high-budget` | 0.04 | Orthonormal RGB DCT units |
| `--seed` | 0 | Random initialization seed |
| `--device` | `auto` | `auto`, `cpu` or `cuda` |
| `--threads` | 4 | PyTorch CPU threads |

The initialization amplitude is 0.001 in coefficient units, projected onto all
constraints. Reproducibility is expected within the same software/device setup;
CUDA and CPU may differ numerically. A zero step count or zero pixel budget
returns the original exactly. Integer PNG bounds use `floor(epsilon)`.

## Saved files

- `<input_stem>_protected.png`: original-size protected image.
- `face_XX_original.png`, `face_XX_protected.png`: aligned face and candidate.
- `face_XX_mask.png`, `face_XX_parsing.png`: spatial mask and semantic classes.
- `face_XX_comparison.png`: aligned comparison and amplified absolute difference.
- `face_XX_saved_comparison.png`: comparison after PNG save and realignment.
- `face_XX_frequency.npz`: clean RGB coefficients, delta, mask, transform and
  normalization metadata. Coefficients have shape `(1, 3, 14, 14, 8, 8)`.
- `metrics.json`: model/input hashes, config, each iterate's loss, aligned and
  saved-image cosine, global pixel Linf, full-image PSNR, runtime and limitations.

Full-image PSNR includes the unchanged background and can therefore be high
even when a small face region changes noticeably.

## Verified run on 2026-10-22

Input: repository `data/example.jpg` (2333x3500, one detected face).
Environment: Windows, Python 3.13.6, PyTorch 2.11.0+cpu, torchvision 0.26.0+cpu,
ONNX 1.18.0, onnx2torch 1.5.15, ONNX Runtime 1.27.0, OpenCV 4.13.0, four CPU threads.
Both configurations used the defaults above and 20 updates.

| Bands | Aligned cosine | Saved-image cosine | Pixel Linf | Runtime (s) |
| --- | ---: | ---: | ---: | ---: |
| High | 0.9951817393 | 0.9958243370 | 8/255 | 10.42 |
| Mid + high | 0.9863223433 | 0.9874874949 | 8/255 | 9.29 |

See [high metrics](privacy_runs/high_metrics.json) and
[mid/high metrics](privacy_runs/mid_high_metrics.json) for the exact recorded values.
Saved-image comparisons: [high](privacy_runs/high_comparison.png) and
[mid + high](privacy_runs/mid_high_comparison.png).
Runtime includes model setup, conversion verification, parsing, optimization and
PNG writing; it is not an edge-device performance benchmark.

Nine regression tests pass, covering legacy output format, OpenCV DCT agreement,
RGB reconstruction, numerical gradient checks, attack loss reduction on a tiny
test-only network, frozen model parameters, frequency/pixel constraints,
reproducibility, no-op cases, inverse-warp identity and invalid arguments.
The real experiment uses pretrained ArcFace; the tiny network is only a unit
test fixture. The original `test.py` was also run successfully.

The similarity decreases are small. No calibrated verification threshold,
recognition failure rate, dataset benchmark, re-detection robustness, JPEG
robustness or independent AdaFace transfer result is claimed. These remain
follow-up evaluation tasks.

## Sources and model terms

- [Original face parsing repository](https://github.com/zllrunning/face-parsing.PyTorch).
- [ArcFace paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html).
- [InsightFace model release](https://github.com/deepinsight/insightface/releases/tag/v0.7).
- [InsightFace alignment template](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py).
- [InsightFace preprocessing](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/arcface_onnx.py).
- [OpenCV YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
- [onnx2torch](https://github.com/ENOT-AutoDL/onnx2torch).
- [PGD reference](https://openreview.net/forum?id=rJzIBfZAb).
- [FRIA paper](https://arxiv.org/abs/2308.05983), motivation only; this is not a
  reproduction of FRIA's source code.

InsightFace code is MIT-licensed. Its pretrained models are limited to
non-commercial research; code licensing does not grant commercial model rights.
YuNet's model license is provided in its upstream model directory. The downloader
uses official sources and checks the following SHA256 values:

```text
w600k_r50.onnx
4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43

face_detection_yunet_2023mar.onnx
8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
```
