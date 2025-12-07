HARD_PATH=/root/PEFT-CL/MLLM/MCITlib

bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json \
    $HARD_PATH/configs/train_configs/LoRA-FT/LLaVA/MLLM-DCL/train/task1.json