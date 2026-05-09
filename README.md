# PKU-AIGI-500K CATC Compression Project

This repository contains a reconstructed CATC compression model for AI-generated
images, along with training, evaluation, and plotting
utilities.

## Basic info

Title: An exploration of prompt text utility in AI-generated image compression

Author: Mariya Pershyna

Abstract: Due to the massive increase in AI-generated data in recent
years, storage and transmission of AI-generated images is a
more pressing problem. One solution is image compression
that is specific to AI-generated images. This project reconstructs CATC, an image compression model for AI-generated
images, and performs a prompt ablation study to understand to
what extent prompt information improves AIGI compression.
The findings are that altering the prompt does reduce image
compression quality by a statistically significant amount, but
this reduction is incredibly small (less than 0.1 dB).

## Requirements

The Python dependencies are listed in:

```powershell
code\requirements.txt
```

The current requirements are:

```text
torch
torchvision
compressai>=1.2
pytorch-msssim
Pillow
numpy
tensorboard
matplotlib
remotezip
git+https://github.com/openai/CLIP.git
```

## Install

From the repository root:

```powershell
cd "C:\Users\masha\OneDrive\Documents\Playground\PKU-AIGI-500K"
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Upgrade `pip`:

```powershell
python -m pip install --upgrade pip
```

Install the project requirements:

```powershell
python -m pip install -r code\requirements.txt
```

If PyTorch/CUDA installation fails or installs the wrong CUDA build, install
`torch` and `torchvision` using the command recommended by the official PyTorch
installer for your GPU/CUDA version, then rerun:

```powershell
python -m pip install -r code\requirements.txt
```

## Sample Training Command

Download the PKU-AIGI-500K dataset from https://huggingface.co/datasets/Forerunner/PKU-AIGI-500K, then put the data paths in the command below. This example trains on the `MJ` subset and writes checkpoints to the path you
provide with `--save-path`.

```powershell
python code\train.py `
  -d "C:\path\to\PKU-AIGI-500K" `
  --save-path "C:\path\to\checkpoints" `
  --log-file "checkpoints\stdout_log.txt" `
  --training-mode step `
  --max-steps 20000 `
  --datasets MJ `
  --batch-size 10 `
  --test-batch-size 10 `
  --lambda 0.05 `
  --patch-size 256 256 `
  --metric mse `
  --lr-step 16000 18500 `
  --log-interval-steps 100 `
  --validation-interval-steps 100 `
  --checkpoint-interval-steps 100 `
  --cuda
```

Replace `C:\path\to\PKU-AIGI-500K` with the dataset root that contains the `MJ`
folder. The training script expects the MJ data to be arranged like this:

```text
C:\path\to\PKU-AIGI-500K\MJ\train
C:\path\to\PKU-AIGI-500K\MJ\val
C:\path\to\PKU-AIGI-500K\MJ\train.txt
C:\path\to\PKU-AIGI-500K\MJ\vaild.txt
```

The validation folder may also be named `valid` or `vaild`, and the validation
prompt file may be named `valid.txt`, `val.txt`, `vaild.txt`, or `MJ.txt`.

## Sample Evaluation Command

Use `eval.py` to evaluate a trained checkpoint on a test image folder and its
matching prompt file. The lambda value is not passed directly to `eval.py`;
instead, choose the checkpoint saved under the folder for the lambda used during
training. 

```powershell
python code\eval.py `
  --checkpoint "C:\path\to\checkpoints\128_0.05\checkpoint_latest.pth.tar" `
  --data-i "C:\path\to\PKU-AIGI-500K\MJ\test" `
  --data-t "C:\path\to\PKU-AIGI-500K\MJ\test.txt" `
  --cuda
```

Replace `0.05` in the checkpoint path with the lambda value you trained with.

