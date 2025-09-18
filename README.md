# EyesMolProject
VLM 기반 Chain-of-Thought를 활용한 광학 화학 구조 인식 < OCSR(Optical Chemical Structure Recognition) >

### 아이젠사이언스 기업과 모두의연구소 아이펠 리서치가 1달간 함께 진행한 프로젝트입니다.
- **멘토** : 아이젠사이언스 **박세정 수석연구원**님
- **팀원** : 13기 팀 EyesMol **김범모**, **서지연**
- 결과물 ( 논문 ) : 추가 예정
- 회고록 : 추가 예정

## Update
- [2025/09/06] LoRA Stacking을 지원하는 코드를 추가하였습니다.
- [2025/09/06] sft tuning 과정에서 eval, eval_loss를 지원하도록 코드를 추가하였습니다.
     - 우리의 eval, eval_loss코드가 원본 레포지토리에 merge 되었습니다. [#183](https://github.com/2U1/Qwen2-VL-Finetune/pull/183)
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
```
VLM을 활용한 2 step 파인튜닝 접근법을 제안합니다.
1) 분자화학 QA데이터를 활용한 Instruction Tuning, 2) tuning된 모델을 활용하여 OCSR CoT tuning을 진행합니다.
PubChem과 ChEBI에서 image-description 총 440K를 수집하고 Gemini2.5 pro를 사용하여 시각적 설명 데이터 180K를 생성했습니다.
약 290K의 단일 유형의 QA 데이터로 첫번재 Instructio Tuning을 진행했고, 모델의 학습이 제대로 되지 않아 원인 분석 후 두번째 Instruction tuning을 하였습니다.
첫번째 모델은 Instruction following 능력을 상실하고 베이스 모델보다 성능이 저하되는 심각한 문제를 발견했습니다.
오류를 수정한 약 660K의 QA 데이터로 Instruction Tuning을 진행했고, 1 step CoT tuning과 2 step CoT tuning을 비교했습니다.
결론적으로, Qwen 2.5 VL 3B 모델에서는 1 step CoT tuning이 2 step CoT tuning보다 성능이 높았습니다.
이 연구는 VLM이 분자화학 이미지를 이해하고 지시를 따르는 능력을 학습하기 위해 필수적으로 고려해야될 점을 제공함과 동시에,
인간의 추론 과정을 모방한 CoT tuning이 OCSR 뿐만 아니라 복잡한 과학 문제 해결에 중요한 역할을 할 수 있음을 보여줍니다.
```

## Prepare Dataset
- 여기서 다운 받을 수 있습니다. [EyesMolDataset](https://huggingface.co/datasets/bbeomdev/EyesMolDataset)

### PubChem 데이터 250K + ChEBI 데이터 180K
- 우리는 PubChem, ChEBI 홈페이지에서 데이터를 수집했습니다.
- ( PubChem은 [MoleculeSTM](https://github.com/chao1224/MoleculeSTM)의 CID2SMILES.CSV 파일 id값을 참고하여 직접 수집했습니다. ChEBI는 [Ontology files](https://www.ebi.ac.uk/chebi/)의 id를 참고하였습니다.)
- PubChem, ChEBI 데이터 제공자의 여러 라이센스로 인해 데이터를 직접 제공하기 어렵습니다. 대신 `instruction_data_id.parquet` 파일의 id를 참조하여 PuBchem, ChEBI의 API를 사용하여 수집하는 것을 권장드립니다.
  
### PubChem Gemini 2.5 pro 이미지 설명 데이터 180K
- `instruction_data_id.parquet` 우리는 Gemini 2.5 pro를 사용하여 PubChem 이미지 180K에 대한 설명 데이터를 생성했습니다.

### Instruction QA 데이터
- 프로젝트 시간, 자원 제약으로 인해 PubChem 데이터만 사용했습니다. ( ChEBI 미포함 )
- Instruction-following 능력을 유지하기 위해 [Pixmo-docs-charts 118k](https://huggingface.co/datasets/allenai/pixmo-docs) 데이터를 섞어서 사용했습니다.
- QA json 형태는 LLaVA 150K 형식을 따릅니다. 자세한 형태는 [LLaVA 15OK](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)를 참고해주세요.
- `instruction_result_v2_train.json`는 저희가 훈련에 사용한 일부 데이터입니다.

### CoT QA 데이터
- [GTR-CoT](https://arxiv.org/abs/2506.07553) 논문의 내용을 참고하였습니다. 공개된 데이터셋과 코드가 없어 완벽하게 일치하지 않으며 x,y 좌표의 경우 다를 수 있습니다.
- MolScribe에서 제공하는 [USPTO 특허 데이터](https://huggingface.co/yujieq/MolScribe/blob/main/uspto_mol.zip)를 사용하여, 단일 분자 구조 이미지만 추출하여 사용했습니다.
- USPTO 데이터셋에 포함된, 각 원자의 좌표(node coordinates)와 원자 간 결합 정보(edges)를 활용하여, Chain-of-Thought (CoT) 과정을 생성했습니다.
- 자세한 코드는 `data_preprocess` 폴더를 참고해주세요
- `cot_result_v2_train.json`는 저희가 훈련에 사용한 일부 데이터입니다.

### images
- 이미지 파일은 특정 폴더 내부에 들어있으면 됩니다.
- ex)
  ```
  - images/
       - image1.png
       - image2.png
  ```

## Train Model
- 여기서 다운 받을 수 있습니다. [EyesMolModel](https://huggingface.co/bbeomdev/EyesMol)

### Instruction QA tuning 코드
- `run_lora_finetuning_insturction.sh` 스크립트 참고

### CoT tuning
- `run_lora_finetuning_cot` 스크립트 참고

### 2 step CoT tuning
- `run_lora_finetuning_cot stack` 스크립트 참고

## Inference
- `inference/simple_inference_cot.ipynb` 참고
- 2 step CoT tuning의 경우 `merge_lora_weights.py` 대신 `merge_lora_stack_weights.py`를 사용하면 됩니다.
- ```
  !python /workspace/EyesMolProject/Qwen2-VL-Finetune/src/merge_lora_stack_weights.py \
     --checkpoint-path {2step_CoT_checkpoint}
     --first-lora-path {instruction_first_lora} \
     --model-base {model_base} \
     --save-model-path {save_path}
  ```

# Citation
```
@misc{EyesMolProject2025,
  title        = {EyesMolProject: VLM 기반 Chain-of-Thought를 활용한 광학 화학 구조 인식},
  author       = {김범모 and 서지연 and 아이젠사이언스 박세정 수석연구원},
  year         = {2025},
  howpublished = {GitHub repository},
  note         = {Apache-2.0 License},
  url          = {https://github.com/bbeomdev/EyesMolProject}
}
```

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

@article{Wang2025GTRCoT,
  title = {GTR-CoT: Graph Traversal as Visual Chain of Thought for Molecular Structure Recognition},
  author = {Jingchao Wang and Haote Yang and Jiang Wu and Yifan He and Xingjian Wei and Yinfan Wang and Chengjin Liu and Lingli Ge and Lijun Wu and Bin Wang and Dahua Lin and Conghui He},
  journal = {arXiv preprint arXiv:2506.07553},
  year = {2025},
  note = {Version v2, submitted June 9, 2025; revised June 10, 2025},
  url = {https://arxiv.org/abs/2506.07553}
}
```
