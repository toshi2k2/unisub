# Adapting RoBERTa to the GLUE Benchmark using SubspaceAdapter

## Overview
Our experiments on the GLUE benchmark are run on 4 NVIDIA A5000 GPU cards. Results may vary due to different GPU models, drivers, CUDA SDK versions, floating-point precisions, and random seeds.

## Steps to Reproduce GLUE Results

### 1. Navigate to the NLU directory
```console
cd NLU/
```

---

## Option A: SubspaceAdapter via Standard SVD (`get_subspaceadapter.py`)

Computes eigenvectors from source LoRA adapters using truncated SVD and generates a SubspaceAdapter initialization for the target task.

### Obtain the initial SubspaceAdapter
You can use the provided shell scripts for each task:
```console
sh ./scripts/generation/get_subspaceadapter_mrpc.sh
sh ./scripts/generation/get_subspaceadapter_stsb.sh
```

Or run the Python script directly (example for MRPC):
```console
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
```

---

## Option B: SubspaceAdapter via HOSVD (`hosvd.py`)

Computes shared subspace bases from multiple source LoRA adapters using Higher-Order SVD (Tucker decomposition) and generates a SubspaceAdapter initialization for the target task.

Two decomposition modes are supported via `--tucker_mode`:
- **mode-2**: Concatenates each layer's LoRA matrices column-wise into a 2D matrix and applies truncated SVD.
- **mode-3**: Stacks each layer's LoRA matrices into a 3-mode tensor and applies full Tucker decomposition.

### Obtain the initial SubspaceAdapter via HOSVD
You can use the provided shell script:
```console
sh ./scripts/generation/get_hosvd.sh
```

Or run the Python script directly (example for MRPC with mode-3 Tucker):
```console
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
```

---

## Training

Once a SubspaceAdapter has been generated (via either Option A or B), start training:
```console
sh ./scripts/training/glue_mrpc.sh
sh ./scripts/training/glue_stsb.sh
```

Or run the training script directly (example for MRPC):
```console
python run_glue.py \
  --model_name_or_path roberta-base \
  --task_name mrpc \
  --do_train \
  --do_eval \
  --max_seq_length 512 \
  --per_device_train_batch_size 16 \
  --seed 0 \
  --learning_rate 4e-3 \
  --lr_scheduler_type 'reduce_lr_on_plateau' \
  --weight_decay 0.1 \
  --warmup_ratio 0.06 \
  --num_train_epochs 30 \
  --evaluation_strategy epoch \
  --apply_subspaceadapter True \
  --subspaceadapter_r 8 \
  --subspaceadapter_num_components 64 \
  --subspaceadapter_adapter_name mrpc \
  --subspaceadapter_load_path ./mrpc_subspaceadapter \
  --subspaceadapter_save_path ./mrpc_subspaceadapter_trained \
  --output_dir ./mrpc \
  --overwrite_output_dir \
  --logging_dir ./mrpc \
  --logging_steps 10 \
  --report_to wandb \
  --run_name roberta_mrpc
```

---

## Evaluation

Evaluate a trained SubspaceAdapter:
```console
python run_glue.py \
  --model_name_or_path roberta-base \
  --task_name mrpc \
  --do_eval \
  --max_seq_length 512 \
  --per_device_train_batch_size 16 \
  --seed 0 \
  --learning_rate 4e-3 \
  --lr_scheduler_type 'reduce_lr_on_plateau' \
  --weight_decay 0.1 \
  --warmup_ratio 0.06 \
  --num_train_epochs 30 \
  --evaluation_strategy epoch \
  --apply_subspaceadapter True \
  --subspaceadapter_r 8 \
  --subspaceadapter_num_components 64 \
  --subspaceadapter_adapter_name mrpc \
  --subspaceadapter_load_path ./mrpc_subspaceadapter_trained \
  --subspaceadapter_save_path ./mrpc_subspaceadapter_trained \
  --output_dir ./mrpc \
  --overwrite_output_dir \
  --logging_dir ./mrpc \
  --logging_steps 10 \
  --report_to wandb \
  --run_name roberta_mrpc
```