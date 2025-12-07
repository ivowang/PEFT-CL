#!/bin/bash

HARD_PATH=/root/PEFT-CL/MLLM/MCITlib


bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-ACL/OCR.json \
    $HARD_PATH/configs/train_configs/MoELoRA/LLaVA/MLLM-ACL/train/task1.json
bash scripts/MCITlib/Eval_MLLM_ACL/Eval_finetune1.sh 1 $HARD_PATH


bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-ACL/Math.json \
    $HARD_PATH/configs/train_configs/MoELoRA/LLaVA/MLLM-ACL/train/task2.json
bash scripts/MCITlib/Eval_MLLM_ACL/Eval_finetune1.sh 2 $HARD_PATH


bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-ACL/VP.json \
    $HARD_PATH/configs/train_configs/MoELoRA/LLaVA/MLLM-ACL/train/task3.json
bash scripts/MCITlib/Eval_MLLM_ACL/Eval_finetune1.sh 3 $HARD_PATH


bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-ACL/APP.json \
    $HARD_PATH/configs/train_configs/MoELoRA/LLaVA/MLLM-ACL/train/task4.json
bash scripts/MCITlib/Eval_MLLM_ACL/Eval_finetune1.sh 4 $HARD_PATH