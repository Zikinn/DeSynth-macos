# macOS 安装与运行指南

本指南适用于 Apple Silicon Mac（M1 或更新型号）。项目使用 Python 3.12
和 PyTorch Metal（MPS）运行，不需要 NVIDIA CUDA。

## 1. 进入项目目录

将下面的路径替换为你实际存放项目的位置：

```bash
cd /path/to/DeSynth-macos
```

## 2. 创建并激活虚拟环境

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
```

激活成功后，终端中使用的 `python` 应指向项目里的 `.venv`：

```bash
which python
python --version
```

## 3. 安装依赖

推荐直接使用仓库提供的依赖文件：

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

等效的手动安装命令如下：

```bash
python -m pip install \
  torch==2.7.1 \
  torchvision==0.22.1 \
  diffusers==0.36.0 \
  'transformers>=4.46,<5' \
  'accelerate>=1,<2' \
  peft==0.17.1 \
  'gguf>=0.10' \
  sentencepiece protobuf pillow safetensors numpy opencv-python scikit-image
```

不要使用原版 DeSynth 的 CUDA 12.8 依赖；CUDA 版本的 PyTorch 不适用于
macOS。

## 4. 确认 PyTorch 和 Apple GPU

```bash
python -c 'import torch; print("torch:", torch.__version__); print("MPS:", torch.backends.mps.is_available())'
```

正常结果应包含：

```text
torch: 2.7.1
MPS: True
```

如果显示 `MPS: False`，请确认：

- 使用的是 Apple Silicon Mac，而不是 Intel Mac。
- 使用的是原生 arm64 Python。
- macOS 已更新到版本 14 或更新版本。
- 当前终端已经激活上面创建的虚拟环境。

## 5. 准备模型文件

模型文件不会包含在 Git 仓库中。运行前，需要把以下两个文件放在项目
根目录：

```text
qwen-image-2512-Q4_K_M.gguf
Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors
```

脚本也兼容首字母大写的 GGUF 文件名
`Qwen-Image-2512-Q4_K_M.gguf`。

## 6. 运行

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python desynth.py \
  path/to/input.png \
  --device mps
```

输出文件会写入项目的 `out/` 目录。使用 `--device mps` 会强制要求
Apple Metal 可用，避免意外以极慢的纯 CPU 模式运行。

第一次运行会从 Hugging Face 下载约 250 MB 的配置和 VAE 文件；后续
运行会使用本地缓存。

## 常见问题

### `ModuleNotFoundError: No module named 'torch'`

虚拟环境没有激活，或者依赖安装到了另一个 Python 环境。重新执行：

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### `ValueError: PEFT backend is required for this method`

安装 LoRA 加载所需的 PEFT：

```bash
python -m pip install peft==0.17.1
```

### 出现 CUDA 不可用警告

macOS 没有 CUDA。GGUF/Diffusers 的部分内部代码可能仍会打印 CUDA
相关警告；只要程序显示 `accelerator: mps`，该警告通常不影响运行。

### 比较原图与输出

```bash
python compare.py path/to/input.png out/path-to-output.png
```

`compare.py` 计算的是图像相似度指标，并不检测 SynthID 或其他水印。
