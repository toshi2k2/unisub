# Mistral LoRAs


## Step 1: Compute SubspaceAdapte
`get_subspaceadapter.py` loads source LoRA adapters, computes principal components via SVD, and produces either **reconstructed LoRA weights** or **SubspaceAdapter components + loadings**.

### Basic usage (reconstruction)

```bash
python get_subspaceadapter.py \
    --source_lora_config train.json \
    --num_components 256 \
    --output_type reconstruction \
    --output_dir ./output
```

### SubspaceAdapter output

```bash
python get_subspaceadapter.py \
    --source_lora_config train.json \
    --target_lora_config eval.json \
    --num_components 256 \
    --output_type subspaceadapter \
    --output_dir ./output
```

### Save / reuse precomputed components

```bash
# Compute and save components
python get_subspaceadapter.py \
    --source_lora_config train.json \
    --save_components \
    --component_output_path ./lola_components.pt \
    --output_dir ./output

# Reuse saved components
python get_subspaceadapter.py \
    --component_path ./lola_components.pt \
    --target_lora_config eval.json \
    --output_dir ./output
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--source_lora_config` | `train.json` | JSON file listing source LoRA adapter names |
| `--target_lora_config` | `None` | JSON file listing target LoRAs (defaults to source) |
| `--target_lora_names` | `None` | Explicit target LoRA names (overrides config) |
| `--model_name_or_path` | *(base model)* | Base causal LM |
| `--num_components` | `256` | Number of SVD components |
| `--lora_r` | `16` | LoRA rank |
| `--component_path` | `None` | Path to precomputed components |
| `--save_components` | `False` | Save computed components to disk |
| `--output_type` | `reconstruction` | `reconstruction` or `subspaceadapter` |
| `--output_dir` | `./output` | Output directory |

---

## Step 2: Evaluate

`mistral_eval.py` evaluates adapters on benchmark tasks. It supports 10 IID tasks and 10 OOD tasks.

### IID tasks

| Task ID | Dataset |
|---|---|
| task076 | correcting_sql_mistake |
| task627 | word_with_same_meaning_sentence_generation |
| task664 | answer_generation_abstract_algebra |
| task819 | sentiment_classification |
| task1631 | answer_generation |
| task852 | synthetic_multiply_odds |
| task1657 | question_generation |
| task879 | schema_guided_classification |
| task1596 | text_generation |
| task382 | answer_generation |

### OOD tasks

| Task ID | Dataset |
|---|---|
| task280 | classification_stereotype_type |
| task190 | nli_classification |
| task391 | causal_relationship |
| task290 | question_answerability |
| task1391 | easy_answer_generation |
| task1342 | reviews_title |
| task442 | paraphrase_question_generation |
| task620 | medical_subject_headings_answer_generation |
| task1598 | long_text_generation |
| task039 | find_overlapping_words |

### Run evaluation

```bash
# Evaluate a single IID task
python mistral_eval.py \
    --task task076 \
    --dataset_source iid \
    --adapter_path ./output/task076_recons \
    --output_dir ./results

# Evaluate a single OOD task
python mistral_eval.py \
    --task task280 \
    --dataset_source ood \
    --adapter_path ./output/task280_recons \
    --output_dir ./results

# Evaluate with a custom dataset
python mistral_eval.py \
    --task mytask \
    --dataset_source custom \
    --custom_dataset <org>/<dataset_name> \
    --adapter_path ./output/mytask_recons \
    --output_dir ./results
```

### Using the shell script

```bash
bash scripts/run_eval.sh --task task076 --dataset_source iid
bash scripts/run_eval.sh --task task280 --dataset_source ood --adapter_path ./output/task280_recons
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--task` | *(required)* | Task identifier (e.g., `task076`) |
| `--dataset_source` | `iid` | `iid`, `ood`, or `custom` |
| `--custom_dataset` | `None` | HuggingFace dataset name (required for `custom`) |
| `--model_name_or_path` | *(base model)* | Base causal LM |
| `--adapter_path` | `./output/{task}` | Path to the adapter directory |
| `--temperature` | `0.1` | Generation temperature |
| `--top_p` | `0.75` | Top-p sampling |
| `--top_k` | `40` | Top-k sampling |
| `--num_beams` | `4` | Beam search beams |
| `--max_new_tokens` | `32` | Max tokens to generate |
| `--batch_size` | `1` | Evaluation batch size |
| `--output_dir` | `./results` | Results output directory |

---

## Step 3: Compute ROUGE-L Scores

`rouge_scorer.py` reads the CSV files produced by `mistral_eval.py` and computes average ROUGE-L F1 scores.

```bash
# Score IID results
python rouge_scorer.py --results_dir ./results/iid

# Score OOD results
python rouge_scorer.py --results_dir ./results/ood

# Save scores to file
python rouge_scorer.py --results_dir ./results/iid --output_file ./results/iid_scores.csv
```

---

## JSON Configuration Files

- **`train.json`** — 10 IID source LoRA adapter paths (used for eigenvector computation).
- **`train_subset.json`** — Same 10 IID adapters (convenience alias).
- **`eval.json`** — 10 OOD target LoRA adapter paths.
- **`lora_list.json`** — Full catalog of 502 LoRA adapters (rank 16, 4-bit).

Each JSON maps an integer index to an adapter path, e.g.:

```json
{
  "0": "<org>/<base_model>-<config>-task280",
  "1": "<org>/<base_model>-<config>-task190"
}
```

---

## End-to-End Example

```bash
# 1. Compute SubspaceAdapter reconstruction for IID tasks
python get_subspaceadapter.py \
    --source_lora_config train.json \
    --num_components 256 \
    --output_type reconstruction \
    --save_components \
    --output_dir ./output

# 2. Evaluate each IID task
for task in task076 task627 task664 task819 task1631 task852 task1657 task879 task1596 task382; do
    python mistral_eval.py \
        --task $task \
        --dataset_source iid \
        --adapter_path ./output/${task}_recons \
        --output_dir ./results
done

# 3. Compute SubspaceAdapter reconstruction for OOD tasks
python get_subspaceadapter.py \
    --component_path ./lola_components.pt \
    --target_lora_config eval.json \
    --num_components 256 \
    --output_type reconstruction \
    --output_dir ./output

# 4. Evaluate each OOD task
for task in task280 task190 task391 task290 task1391 task1342 task442 task620 task1598 task039; do
    python mistral_eval.py \
        --task $task \
        --dataset_source ood \
        --adapter_path ./output/${task}_recons \
        --output_dir ./results
done

# 5. Compute ROUGE-L scores
python rouge_scorer.py --results_dir ./results/iid
python rouge_scorer.py --results_dir ./results/ood
```
