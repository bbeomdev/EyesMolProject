# EyesMolProject

A molecular visualization and analysis project.

## Installation
### Environments
- Ubuntu 22.04
- Cuda version 12.4
- Using uv and requirements.txt

1. Install uv:
   ```bash
   pip install uv
   ```

2. Create virtual environment with Python 3.11:
   ```bash
   uv venv venv --python 3.11
   ```
   - Enter the virtual environment after creation ( source venv/bin/activate )

3. Install requirements with PyTorch CUDA 12.4:
   ```bash
   uv pip install -r requirements.txt -f https://download.pytorch.org/whl/cu124
   ```

4. Install flash-attn:
   ```bash
   uv pip install flash-attn==2.7.4.post1 --no-build-isolation
   ```

5. Update system packages:
   ```bash
   apt update
   ```

6. Install git-lfs:
   ```bash
   apt install git-lfs
   ```

7. Clone Qwen2.5-VL-3B-Instruct model:
   ```bash
   git lfs clone https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
   ```

