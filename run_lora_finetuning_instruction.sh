cd /workspace/Qwen2-VL-Finetune

export PYTHONPATH=src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Model configuration
MODEL_PATH="/workspace/Qwen2.5-VL-3B-Instruct"
DATA_PATH="/workspace/single_turn_qa_data.json"
IMAGE_FOLDER="/workspace/molecule_images"
OUTPUT_DIR="output/qwen_3B_Qlora_train"

# Wandb 설정
export WANDB_PROJECT="qwen2.5-vl-3b-qlora_train"
export WANDB_RUN_NAME="qlora-4bit-$(date +%Y%m%d-%H%M%S)"
export WANDB_LOG_MODEL="checkpoint"  # 체크포인트 자동 업로드
export WANDB_WATCH="all"  # 그래디언트와 파라미터 추적

# Wandb API 키 설정 (필요한 경우)
export WANDB_API_KEY=""

# 또는 wandb login을 통해 인증
echo "Wandb 로그인 확인 중..."
wandb login --verify

# A6000 4개 사용
GLOBAL_BATCH_SIZE=64
BATCH_PER_DEVICE=4
NUM_DEVICES=4
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

mkdir -p $OUTPUT_DIR
echo "Starting LoRA fine-tuning..."

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
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero2.json \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
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
    --save_steps 1000 \
    --save_total_limit 3 \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --group_by_length False \
    --optim adamw_bnb_8bit \
    --image_resized_width 300 \
    --image_resized_height 300 \
    --resume_from_checkpoint $OUTPUT_DIR
