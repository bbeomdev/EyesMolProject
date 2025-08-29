# EyesMolProject
### 아이젠사이언스 기업과 모두의연구소 아이펠 리서치가 1달간 함께 진행한 프로젝트입니다.
- 멘토 : 아이젠사이언스 박세정 수석연구원님
- 팀원 : 13기 김범모, 서지연
- 결과물 ( 논문 ) : 추가 예정
- 모델 가중치 : 추가 예정

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
   
## Prepare Dataset
### Pubchem 데이터 25만개
- .
### ChEBI 데이터 18만개
- .
### Pubchem Instruction QA dataset 290만
- .
### ChEBI Instruction QA dataset

## Instruction QA tuning 코드
- 내용 추가 예정

## Instruction tuning 모델 inference 노트북
- https://github.com/bbeomdev/EyesMolProject/blob/main/inference.ipynb

## CoT tuning 코드
- 내용 추가 예정



