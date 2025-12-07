# #!/bin/bash

HARD_PATH=/root/PEFT-CL/MLLM/MCITlib

# 
# bash scripts/MCITlib/Eval_MLLM_ACL/eval_OCR.sh $HARD_PATH/configs/modal_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-ACL/OCR.json $HARD_PATH/configs/train_configs/LoRA-FT/InternVL/MLLM-ACL/eval/zero_shot.json
# 
# bash scripts/MCITlib/Eval_MLLM_ACL/eval_Math.sh $HARD_PATH/configs/modal_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-ACL/Math.json $HARD_PATH/configs/train_configs/LoRA-FT/InternVL/MLLM-ACL/eval/zero_shot.json
# 
# bash scripts/MCITlib/Eval_MLLM_ACL/eval_VP.sh $HARD_PATH/configs/modal_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-ACL/VP.json $HARD_PATH/configs/train_configs/LoRA-FT/InternVL/MLLM-ACL/eval/zero_shot.json

bash scripts/MCITlib/Eval_MLLM_ACL/eval_APP.sh $HARD_PATH/configs/modal_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-ACL/APP.json $HARD_PATH/configs/train_configs/LoRA-FT/InternVL/MLLM-ACL/eval/zero_shot.json