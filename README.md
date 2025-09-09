# EyesMolProject
### 아이젠사이언스 기업과 모두의연구소 아이펠 리서치가 1달간 함께 진행한 프로젝트입니다.
- **멘토** : 아이젠사이언스 **박세정 수석연구원**님
- **팀원** : 13기 **김범모**, **서지연**
- 결과물 ( 논문 ) : 추가 예정
- 회고록 : 추가 예정

## Update
- [2025/09/06] LoRA Stacking을 지원하는 코드를 추가하였습니다. ( 우리의 2step CoT train 과정을 참고해주세요. )
- [2025/09/06] sft tuning 과정에서 eval, eval_loss를 지원하도록 코드를 추가하였습니다.
     - 현재 원본 레포지토리에 PR 예정입니다.
- [2025/08/28] Qwen2.5 VL, QLoRA, gradient_checkpointing을 같이 쓰면 발생하는 오류를 해결하였습니다.
     - 우리의 버그 해결 코드가 원본 레포지토리에 merge 되었습니다. [#178](https://github.com/2U1/Qwen2-VL-Finetune/pull/178)

## Environments
- 2U1 / [Qwen2-VL-Finetune](https://github.com/2U1/Qwen2-VL-Finetune) 레포지토리를 기반으로 제작되었습니다.
- Ubuntu 22.04
- Cuda version 12.4
- Using uv and requirements.txt

### Installation

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
4. Install wandb
   ```bash
   uv pip install wandb
   wandb login <your_api_key>

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
   
## Abstract
- 

## Prepare Dataset
[EyesMolDataset](https://huggingface.co/datasets/bbeomdev/EyesMolDataset)

### PubChem 데이터 250K + ChEBI 데이터 180K
- 우리는 PubChem, ChEBI 홈페이지에서 데이터를 수집했습니다.
- ( PubChem은 [MoleculeSTM](https://github.com/chao1224/MoleculeSTM)의 CID2SMILES.CSV 파일 id값을 참고하여 직접 수집했습니다. ChEBI는 [Ontology files](https://www.ebi.ac.uk/chebi/)의 id를 참고하였습니다.)
- PubChem, ChEBI 데이터 제공자의 여러 라이센스로 인해 데이터를 직접 제공하기 어렵습니다. 대신 `instruction_data_id.parquet` 파일의 id를 참조하여 `utils/scrape_data.ipynb`로 수집할 수 있습니다.
  
### PubChem Gemini 2.5 pro 이미지 설명 데이터 180K
- `instruction_data_id.parquet` 우리는 Gemini 2.5 pro를 사용하여 PubChem 이미지 180K에 대한 설명 데이터를 생성했습니다.

### Instruction QA 데이터
- 프로젝트 시간, 자원 제약으로 인해 PubChem 데이터만 사용했습니다. ( ChEBI 미포함 )
- Instruction-following 능력을 유지하기 위해 [Pixmo-docs-charts 118k](https://huggingface.co/datasets/allenai/pixmo-docs) 데이터를 섞어서 사용했습니다.
- 자세한 코드는 `utils/qa_generator.ipynb`를 참고해주세요.

### CoT QA 데이터
- MolScribe에서 제공하는 [USPTO 특허 데이터](https://huggingface.co/yujieq/MolScribe/blob/main/uspto_mol.zip)를 사용하여, 단일 분자 구조 이미지만 추출하여 사용했습니다.
- USPTO 데이터셋에 포함된, 각 원자의 좌표(node coordinates)와 원자 간 결합 정보(edges)를 활용하여, Chain-of-Thought (CoT) 과정을 생성
- 자세한 코드는 `data_preprocess` 폴더를 참고해주세요


### Pubchem Instruction QA dataset 290만
- .

## Train Model
- [EyesMolModel](https://huggingface.co/bbeomdev/EyesMol)

### Instruction QA tuning 코드
- 필수 

### CoT tuning
- 필수


## Inference
- 

## Instruction tuning 모델 inference 노트북
- [https://github.com/bbeomdev/EyesMolProject/blob/main/inference.ipynb](https://github.com/bbeomdev/EyesMolProject/blob/main/inference/inference.ipynb)

## CoT tuning 코드
- 내용 추가 예정


# Citation
```
@misc{Qwen2-VL-Finetuning,
  author = {Yuwon Lee},
  title = {Qwen2-VL-Finetune},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/2U1/Qwen2-VL-Finetune}
}

@article{molmo2024,
  title={Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models},
  author={Matt Deitke and Christopher Clark and Sangho Lee and Rohun Tripathi and Yue Yang and Jae Sung Park and Mohammadreza Salehi and Niklas Muennighoff and Kyle Lo and Luca Soldaini and Jiasen Lu and Taira Anderson and Erin Bransom and Kiana Ehsani and Huong Ngo and YenSung Chen and Ajay Patel and Mark Yatskar and Chris Callison-Burch and Andrew Head and Rose Hendrix and Favyen Bastani and Eli VanderBilt and Nathan Lambert and Yvonne Chou and Arnavi Chheda and Jenna Sparks and Sam Skjonsberg and Michael Schmitz and Aaron Sarnat and Byron Bischoff and Pete Walsh and Chris Newell and Piper Wolters and Tanmay Gupta and Kuo-Hao Zeng and Jon Borchardt and Dirk Groeneveld and Jen Dumas and Crystal Nam and Sophie Lebrecht and Caitlin Wittlif and Carissa Schoenick and Oscar Michel and Ranjay Krishna and Luca Weihs and Noah A. Smith and Hannaneh Hajishirzi and Ross Girshick and Ali Farhadi and Aniruddha Kembhavi},
  journal={arXiv preprint arXiv:2409.17146},
  year={2024}
}

@article{liu2023moleculestm,
    title={Multi-modal molecule structure-text model for text-based retrieval and editing},
    author={Liu, Shengchao and Nie, Weili and Wang, Chengpeng and Lu, Jiarui and Qiao, Zhuoran and Liu, Ling and Tang, Jian and Xiao, Chaowei and Anandkumar, Anima},
    title={Multi-modal molecule structure--text model for text-based retrieval and editing},
    journal={Nature Machine Intelligence},
    year={2023},
    month={Dec},
    day={01},
    volume={5},
    number={12},
    pages={1447-1457},
    issn={2522-5839},
    doi={10.1038/s42256-023-00759-6},
    url={https://doi.org/10.1038/s42256-023-00759-6}
}

@article{Qwen2.5-VL,
  title={Qwen2.5-VL Technical Report},
  author={Bai, Shuai and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Song, Sibo and Dang, Kai and Wang, Peng and Wang, Shijie and Tang, Jun and Zhong, Humen and Zhu, Yuanzhi and Yang, Mingkun and Li, Zhaohai and Wan, Jianqiang and Wang, Pengfei and Ding, Wei and Fu, Zheren and Xu, Yiheng and Ye, Jiabo and Zhang, Xi and Xie, Tianbao and Cheng, Zesen and Zhang, Hang and Yang, Zhibo and Xu, Haiyang and Lin, Junyang},
  journal={arXiv preprint arXiv:2502.13923},
  year={2025}
}

@article{Qwen2-VL,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Wang, Peng and Bai, Shuai and Tan, Sinan and Wang, Shijie and Fan, Zhihao and Bai, Jinze and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Fan, Yang and Dang, Kai and Du, Mengfei and Ren, Xuancheng and Men, Rui and Liu, Dayiheng and Zhou, Chang and Zhou, Jingren and Lin, Junyang},
  journal={arXiv preprint arXiv:2409.12191},
  year={2024}
}

@article{Qwen-VL,
  title={Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond},
  author={Bai, Jinze and Bai, Shuai and Yang, Shusheng and Wang, Shijie and Tan, Sinan and Wang, Peng and Lin, Junyang and Zhou, Chang and Zhou, Jingren},
  journal={arXiv preprint arXiv:2308.12966},
  year={2023}
}
```
