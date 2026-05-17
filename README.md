# 对抗扰动驱动的云边协同人脸隐私保护

本项目为大学生创新创业训练计划项目，围绕云边协同场景下的人脸隐私保护问题展开。当前代码以人脸语义解析为基础模块，基于 PyTorch 实现人脸部件分割、解析结果可视化和局部区域处理演示，为后续引入对抗扰动、生物特征脱敏和云边协同部署提供基础。

项目参考 BiSeNet 语义分割网络，并使用 CelebAMask-HQ 数据集进行训练与测试。模型可以将人脸图像中的眉毛、眼睛、鼻子、嘴唇、头发、皮肤等区域解析为不同语义类别，为后续的人脸隐私区域定位、对抗扰动生成和隐私保护效果评估提供基础。

## 项目来源与二次开发说明

本项目是在开源项目 [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) 的基础上开展的大学生创新创业训练计划项目。

原项目主要实现了基于 BiSeNet 的人脸语义解析功能，并提供了 CelebAMask-HQ 数据预处理、模型训练、推理可视化和局部妆容示例等代码。本项目保留其人脸解析基础能力，并围绕“对抗扰动驱动的云边协同人脸隐私保护”这一大创课题进行整理和后续扩展。

原项目地址：

- [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)
- [原项目 modules/src 目录](https://github.com/zllrunning/face-parsing.PyTorch/tree/master/modules/src)

## 项目功能

- 人脸语义解析：对输入人脸图像进行像素级分类。
- 解析结果可视化：将预测的语义分割结果叠加到原图上。
- 核心五官区域提取：在 Demo 中保留眉毛、眼睛、鼻子、嘴唇等核心区域。
- 局部妆容演示：基于解析结果对头发、嘴唇等区域进行换色。

## 项目进度清单

- [x] 完成人脸语义解析基础代码整理。

- [ ] 待补充


## 目录结构

```text
.
├── README.md              # 项目说明文档
├── train.py               # 模型训练入口
├── test.py                # Demo 推理入口
├── evaluate.py            # 训练过程中的推理可视化脚本
├── prepropess_data.py     # CelebAMask-HQ 标签预处理脚本
├── face_dataset.py        # 数据集读取与增强
├── transform.py           # 数据增强方法
├── model.py               # BiSeNet 模型结构
├── resnet.py              # ResNet18 backbone
├── loss.py                # OHEM 交叉熵损失
├── optimizer.py           # SGD 优化器与学习率策略
├── makeup.py              # 局部妆容效果演示
├── data/                  # Demo 输入图片目录
├── res/cp/                # 模型权重目录
└── res/test_res/          # Demo 输出结果目录
```

## 环境要求

建议使用 Python 虚拟环境安装依赖，避免把依赖包直接安装到系统环境中。

创建并进入虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
```

安装项目依赖：

```bash
pip install -r requirements.txt
```

如果使用的是 Windows PowerShell，可以用下面的方式进入虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

主要依赖包括：

- Python
- PyTorch
- torchvision
- NumPy
- OpenCV
- Pillow
- scikit-image
- tqdm

当前推理和训练代码默认使用 GPU，并在代码中调用了 `.cuda()`。如果在 CPU 环境运行，需要额外修改模型和张量的设备逻辑。

`requirements.txt` 来自当前开发环境，包含 PyTorch、torchvision、OpenCV、Pillow、NumPy 以及 CUDA 相关依赖。不同显卡、CUDA 版本或操作系统下，PyTorch 相关依赖可能需要根据本机环境调整。

## 快速运行 Demo

仓库中已包含示例输入图片和模型权重：

- 输入图片：`data/example.jpg`
- 模型权重：`res/cp/79999_iter.pth`

运行 Demo：

```bash
python test.py
```

如果使用仓库内虚拟环境，可以运行：

```bash
.venv/bin/python test.py
```

运行后，结果会保存到：

```text
res/test_res/
```

其中：

- `.jpg` 文件为分割结果叠加到原图后的可视化结果。
- `.png` 文件为经过核心五官类别过滤后的语义 mask。

## 数据预处理

训练前需要下载 CelebAMask-HQ 数据集，并将原始的分散部件 mask 合成为单张语义标签图。

预处理脚本：

```bash
python prepropess_data.py
```

运行前需要修改 `prepropess_data.py` 中的数据路径：

```python
face_data = '/home/zll/data/CelebAMask-HQ/CelebA-HQ-img'
face_sep_mask = '/home/zll/data/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
mask_path = '/home/zll/data/CelebAMask-HQ/mask'
```

标签类别共 19 类，其中 `0` 表示背景，`1` 到 `18` 表示不同人脸部件：

```text
1  skin
2  l_brow
3  r_brow
4  l_eye
5  r_eye
6  eye_g
7  l_ear
8  r_ear
9  ear_r
10 nose
11 mouth
12 u_lip
13 l_lip
14 neck
15 neck_l
16 cloth
17 hair
18 hat
```

## 模型训练

训练入口为：

```bash
python train.py
```

但当前训练脚本按分布式训练方式编写，README 原始运行方式为：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
```

新版本 PyTorch 也可以使用 `torchrun`：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py
```

训练前需要重点检查 `train.py` 中的配置：

```python
data_root = '/home/zll/data/CelebAMask-HQ/'
n_classes = 19
n_img_per_gpu = 16
cropsize = [448, 448]
max_iter = 80000
```

训练过程中，每 5000 次迭代会保存一次模型权重到：

```text
res/cp/
```

训练结束后会保存最终模型到：

```text
res/model_final_diss.pth
```

## 推理说明

`test.py` 是当前主要 Demo 脚本。其默认行为为：

1. 从 `res/cp/79999_iter.pth` 加载模型权重。
2. 遍历 `data/` 目录下的图片。
3. 将图片缩放到 `512 x 512`。
4. 使用 BiSeNet 输出语义分割结果。
5. 只保留核心五官相关类别。
6. 将结果保存到 `res/test_res/`。

核心五官类别在 `test.py` 中定义：

```python
CORE_FACE_PART_CLASSES = {2, 3, 4, 5, 10, 11, 12, 13}
```

如果需要保留头发、皮肤、脖子、衣服等完整类别，可以根据需求修改该集合，或者使用 `evaluate.py` 中不过滤类别的可视化逻辑。

## 妆容效果演示

`makeup.py` 基于已经生成的语义 mask 对局部区域进行换色，例如头发和嘴唇。

默认示例中使用的类别包括：

```python
hair = 17
upper_lip = 12
lower_lip = 13
```

该脚本中仍包含作者本机路径和 GUI 显示逻辑，运行前需要根据本地环境修改图片路径、mask 路径和输出路径。

## 注意事项

- 当前代码中存在多处硬编码路径，训练和预处理前需要改成自己的数据集路径。
- `test.py` 和 `evaluate.py` 都是推理可视化脚本，不计算 mIoU、pixel accuracy 等正式评估指标。
- `train.py` 默认按多 GPU 分布式训练设计，单 GPU 或 CPU 环境需要修改启动方式和设备逻辑。
- `resnet.py` 初始化时会加载 ResNet18 预训练权重，离线环境需要提前准备本地缓存。
- `prepropess_data.py` 文件名保留了原项目拼写，实际作用是数据预处理。

## 参考

- [BiSeNet](https://github.com/CoinCheung/BiSeNet)
- [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)
- [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)
