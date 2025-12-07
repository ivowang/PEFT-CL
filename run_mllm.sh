#!/bin/bash

METHOD=""
TASK=""

while [ $# -gt 0 ]; do
    case $1 in
        --method)
            METHOD="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Usage: sh run_mllm.sh --method <method_name> --task <task_name>"
            echo "Example: sh run_mllm.sh --method HiDe --task DCL"
            exit 1
            ;;
    esac
done

if [ -z "$METHOD" ] || [ -z "$TASK" ]; then
    echo "Error: Missing required parameters"
    echo "Usage: sh run_mllm.sh --method <method_name> --task <task_name>"
    echo "Example: sh run_mllm.sh --method HiDe --task DCL"
    exit 1
fi

BASE_DIR="/root/PEFT-CL"
METHOD_DIR="$BASE_DIR/MLLM/MCITlib/LLaVA/$METHOD"
SCRIPT_PATH="$METHOD_DIR/scripts/MCITlib/Train/train_${TASK}.sh"

if [ ! -d "$METHOD_DIR" ]; then
    echo "Error: Method directory does not exist: $METHOD_DIR"
    echo "Available methods: HiDe, LoRA-FT, DISCO, CL-MoE, MoELoRA, OLoRA, SEFE, ModalPrompt"
    exit 1
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script file does not exist: $SCRIPT_PATH"
    echo "Please check if the task name is correct"
    exit 1
fi

echo "Executing method: $METHOD"
echo "Executing task: $TASK"
echo "Script path: $SCRIPT_PATH"
echo "----------------------------------------"

cd "$METHOD_DIR" && sh "scripts/MCITlib/Train/train_${TASK}.sh"

