#!/bin/bash

################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-7b-v1.5"
################## VICUNA ##################

MODEL_CONFIG=$1
DATA_CONFIG=$2

read_config() {
    python3 -c "import json; print(json.load(open('$1'))['$2'])"
}

GPU_NUM=$(read_config "$MODEL_CONFIG" gpu_num)
RANK=$(read_config "$MODEL_CONFIG" rank)
MODEL_NAME=$(read_config "$MODEL_CONFIG" model_name)
PREVIOUS=$(read_config "$MODEL_CONFIG" previous_model)
DATA_PATH=$(read_config "$DATA_CONFIG" train_path)
IMAGE=$(read_config "$DATA_CONFIG" image_folder)
VISION_TOWER=$(read_config "$MODEL_CONFIG" vision_tower)
OUTPUT_DIR=$(read_config "$MODEL_CONFIG" output_dir)
EPOCH=$(read_config "$MODEL_CONFIG" epoch)
BATCH_SIZE=$(read_config "$MODEL_CONFIG" batch_size)
GRAD_ACC=$(read_config "$MODEL_CONFIG" grad_acc)
LR=$(read_config "$MODEL_CONFIG" lr)
SPACE_PATH=$(read_config "$MODEL_CONFIG" space_path)
INIT_alpha=$(read_config "$MODEL_CONFIG" init_alpha)
INIT_beta=$(read_config "$MODEL_CONFIG" init_beta)

GPU_LIST=""
for i in $(seq 0 $((GPU_NUM-1))); do
    GPU_LIST+="$i,"
done
GPU_LIST=${GPU_LIST%,}


torchrun --nproc_per_node=4  --master_port 9001 llava/train/train_mem.py \
    --lora_enable True --lora_r $RANK --lora_alpha $((RANK * 5)) --mm_projector_lr 2e-5 \
    --freeze_lora_A True \
    --init_lora_from_gradients ${OUTPUT_DIR}_gradients/gradients_merged.pt \
    --model_name_or_path $MODEL_NAME \
    --previous_task_model_path $PREVIOUS \
    --version $PROMPT_VERSION \
    --data_path $DATA_PATH \
    --image_folder $IMAGE \
    --vision_tower $VISION_TOWER \
    --space_path $SPACE_PATH \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps $GRAD_ACC \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --ddp_find_unused_parameters False \
    --init_alpha $INIT_alpha \
    --init_beta $INIT_beta
