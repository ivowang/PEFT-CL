# #!/bin/bash

TASK_ID=$1
HARD_PATH=/root/PEFT-CL/MLLM/MCITlib

if [ "$TASK_ID" == "1" ]; then
    
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json $HARD_PATH/configs/train_configs/SEFE/LLaVA/MLLM-DCL/eval/task1.json
elif [ "$TASK_ID" == "2" ]; then
    
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json $HARD_PATH/configs/train_configs/SEFE/LLaVA/MLLM-DCL/eval/task2.json
    
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json $HARD_PATH/configs/train_configs/SEFE/LLaVA/MLLM-DCL/eval/task2.json
elif [ "$TASK_ID" == "3" ]; then
    
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json $HARD_PATH/configs/train_configs/SEFE/LLaVA/MLLM-DCL/eval/task3.json
    
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json $HARD_PATH/configs/train_configs/SEFE/LLaVA/MLLM-DCL/eval/task3.json
    
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_ad.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/AD.json $HARD_PATH/configs/train_configs/SEFE/LLaVA/MLLM-DCL/eval/task3.json
elif [ "$TASK_ID" == "4" ]; then
    
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_ad.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/AD.json $HARD_PATH/configs/train_configs/SEFE/LLaVA/MLLM-DCL/eval/task4.json
    
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json $HARD_PATH/configs/train_configs/SEFE/LLaVA/MLLM-DCL/eval/task4.json
    
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json $HARD_PATH/configs/train_configs/SEFE/LLaVA/MLLM-DCL/eval/task4.json
    
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_sci.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/Sci.json $HARD_PATH/configs/train_configs/SEFE/LLaVA/MLLM-DCL/eval/task4.json
else
    
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_ad.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/AD.json $HARD_PATH/configs/train_configs/SEFE/LLaVA/MLLM-DCL/eval/task5.json
    
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json $HARD_PATH/configs/train_configs/SEFE/LLaVA/MLLM-DCL/eval/task5.json
    
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json $HARD_PATH/configs/train_configs/SEFE/LLaVA/MLLM-DCL/eval/task5.json
    
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_sci.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/Sci.json $HARD_PATH/configs/train_configs/SEFE/LLaVA/MLLM-DCL/eval/task5.json
    
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_fin.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/Fin.json $HARD_PATH/configs/train_configs/SEFE/LLaVA/MLLM-DCL/eval/task5.json
fi