#!/bin/bash

# Script to compute subspaceadapter components for MRPC task
# Uses CoLA and QNLI as source LoRA adapters

python get_subspaceadapter.py \
  --source_lora_paths "./cola_lora" "./qnli_lora" \
  --source_lora_names cola qnli \
  --target_task_name mrpc \
  --num_labels 2 \
  --model_name_or_path roberta-base \
  --subspaceadapter_r 8 \
  --num_eigenvector_components 32 \
  --num_gram_schmidt_components 32 \
  --loading_source_index 0 \
  --output_dir ./mrpc_subspaceadapter \
  --save_subspaceadapter_components \
  --save_subspaceadapter_loadings
