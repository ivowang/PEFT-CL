#!/bin/bash

HARD_PATH=/root/PEFT-CL/MLLM/MCITlib


pip install -e ".[train]"
bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json \
    $HARD_PATH/configs/train_configs/ModalPrompt/LLaVA/UCIT/train/task1.json


pip install -e ".[train]"
bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/ArxivQA.json \
    $HARD_PATH/configs/train_configs/ModalPrompt/LLaVA/UCIT/train/task2.json


pip install -e ".[train]"
bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/VizWiz.json \
    $HARD_PATH/configs/train_configs/ModalPrompt/LLaVA/UCIT/train/task3.json


pip install -e ".[train]"
bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/IconQA.json \
    $HARD_PATH/configs/train_configs/ModalPrompt/LLaVA/UCIT/train/task4.json


pip install -e ".[train]"
bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/CLEVR-Math.json \
    $HARD_PATH/configs/train_configs/ModalPrompt/LLaVA/UCIT/train/task5.json


pip install -e ".[train]"
bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/Flickr30k.json \
    $HARD_PATH/configs/train_configs/ModalPrompt/LLaVA/UCIT/train/task6.json