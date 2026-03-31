#!/bin/bash
# Sample script: compute a SubspaceAdapter init for MRPC via HOSVD,
# using six GLUE LoRA adapters as the source tensor basis.

python hosvd.py \
  --source_lora_paths \
    ./cola_lora_r_8 \
    ./qnli_lora_r_8 \
    ./mrpc_lora_r_8 \
    ./rte_lora_r_8 \
    ./sst2_lora_r_8 \
    ./stsb_lora_r_8 \
  --source_lora_names cola qnli mrpc rte sst2 stsb \
  --target_task_name mrpc \
  --num_labels 2 \
  --model_name_or_path roberta-base \
  --subspaceadapter_r 8 \
  --num_hosvd_components 32 \
  --tucker_mode 3 \
  --num_gram_schmidt_components 0 \
  --loading_source_index 2 \
  --output_dir ./mrpc_hosvd_subspaceadapter
