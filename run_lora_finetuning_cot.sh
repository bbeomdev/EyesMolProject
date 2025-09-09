cd /workspace/EyesMolProject/Qwen2-VL-Finetune

export PYTHONPATH=src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Model configuration
MODEL_PATH="/workspace/Qwen2.5-VL-3B-Instruct"
DATA_PATH="/workspace/cot_data_V2/uspto_cot_llava_train_V2.json"
IMAGE_FOLDER="/workspace/images"
OUTPUT_DIR="output/qwen_3B_Qlora_cot_train"
EVAL_PATH="/workspace/cot_data_V2/uspto_cot_llava_valid_V2.json"

# Wandb 설정
export WANDB_PROJECT="qwen2.5-vl-3b-qlora_cot_train"
export WANDB_RUN_NAME="qlora-4bit-$(date +%Y%m%d-%H%M%S)"
export WANDB_LOG_MODEL="checkpoint"  # 체크포인트 자동 업로드
export WANDB_WATCH="all"  # 그래디언트와 파라미터 추적

# Wandb API 키 설정 (필요한 경우)
export WANDB_API_KEY="YOUR_API_KEY"

# 또는 wandb login을 통해 인증
echo "Wandb 로그인 확인 중..."
wandb login --verify

# A6000 4개 사용 -> lora rank를 올렸기 때문에 batch 4-> 2로 수정
GLOBAL_BATCH_SIZE=64
BATCH_PER_DEVICE=2
NUM_DEVICES=4
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

mkdir -p $OUTPUT_DIR
echo "Starting LoRA fine-tuning..."

#CoT 파인튜닝에서는 rank는 높이고, alpha는 적절히 조정하
#alpha/rank 비율 2.0~2.5 유지
# 높은 rank에서는 학습률 약간 낮춤
#CoT는 출력 패턴 변화가 중요 → lm_head 포함이 도움

# Run training with DeepSpeed
deepspeed src/train/train_sft.py \
    --model_id $MODEL_PATH \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((512 * 28 * 28)) \
    --use_liger False \
    --bits 4 \
    --lora_enable True \
    --vision_lora True \
    --use_dora False \
    --lora_namespan_exclude "['embed_tokens']"\
    --lora_rank 64\
    --lora_alpha 128\
    --lora_dropout 0.1\
    --num_lora_modules -1 \
    --deepspeed scripts/zero2.json \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --num_train_epochs 1 \
    --learning_rate 1.5e-4 \
    --merger_lr 1e-5 \
    --vision_lr 1e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to wandb \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --group_by_length False \
    --optim adamw_bnb_8bit \
    --eval_steps 500 \
    --eval_strategy "steps" \
    --per_device_eval_batch_size $BATCH_PER_DEVICE \
    --eval_path $EVAL_PATH \
