# Deliberative Alignment: Reasoning Enables Safer Language Models

> _"To have a profoundly ethical AI system it also has to be very capable.It need a really good world model, a good understanding of ethics and it need really god reasoning. Because if you don't have any of those things How can you possibly be consistently profondly ethical ? - Shane Legg Cofouder of deepmind"_

Welcome to the repository—a reimplementation of the paper [Deliberative Alignment: Reasoning Enables Safer Language Models](https://arxiv.org/abs/2412.16339). This project demonstrates how to train language models, specifically using the **DeepSeek-R1** model, so the model can explicitly reason over its own safety policies before producing a response. The approach enhances safety by ensuring that each generated answer complies with a well-defined set of reasoning steps and policy constraints.

---

## Repository Overview

This repository is organized into several modules, each implementing a critical stage of the training pipeline:

- **`main.py`**  
  The entry point that orchestrates the training pipeline. It supports three stages:
  1. **Filtering Stage:** Data generation and safety evaluation via supervised fine-tuning data extraction.
  2. **Supervised Fine-Tuning (SFT):** Fine-tuning the model on the generated and filtered dataset.
  3. **Reinforcement Learning (RL):** Refinement of the SFT model using a PPO-based RL loop.
  
- **`config/`**  
  - **`config.yaml`**: Contains primary configuration parameters (e.g., model names, training hyperparameters, and dataset details).
  - **`policies.yaml`**: Defines the safety policy rules and response styles which the model must abide by.
  
- **`sft_data_generation.py`**  
  Implements data extraction and augmentation using safety policies and generates training examples with explicit chain-of-thought reasoning.
  
- **`sft.py`**  
  Sets up and runs the supervised fine-tuning stage on a single GPU, leveraging LoRA for efficient training.
  
- **`rl.py`**  
  Carries out reinforcement learning using a PPO Trainer from the TRL library. The RL stage further refines the SFT model by evaluating responses against safety policies.
  
- **`evaluate.py`**  
  Provides a simple evaluation script that tests the trained model on the XSTest dataset, checking if the model produces helpful answers for safe queries and appropriate safe completions (refusals) for unsafe queries.

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/atonkamanda/delibera.git
   cd DeliberativeAlignment
   ```

2. **Set up a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   Make sure you have [pip](https://pip.pypa.io/en/stable/) installed, then run:

   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** The `requirements.txt` includes all necessary dependencies such as `torch`, `transformers`, `datasets`, `trl`, `pyyaml`, and others.

---

## Usage

The training pipeline consists of three major stages—filtering, supervised fine-tuning (SFT), and reinforcement learning (RL). Each stage can be run individually or as part of the full pipeline.

### 1. Pipeline Execution

To run the **entire pipeline** (filtering → SFT → RL) on a single GPU, execute:
```bash
python main.py --config config/config.yaml --stage pipeline
```

### 2. Stage-specific Execution

- **Filtering Stage (Data Generation & Safety Alignment):**

  ```bash
  python main.py --config config/config.yaml --stage filter
  ```

- **Supervised Fine-Tuning (SFT):**

  ```bash
  python main.py --config config/config.yaml --stage sft
  ```

- **Reinforcement Learning (RL):**

  ```bash
  python main.py --config config/config.yaml --stage rl
  ```

Each command automatically loads configurations from `config/config.yaml` (which in turn references the policy file `config/policies.yaml`) and sets up the environment for training on a single GPU. The filtering stage processes data from the WildChat-1M dataset, while the SFT and RL stages fine-tune and refine the model respectively.

---

## Evaluation

After training, you can evaluate the performance of your RL-refined model using the evaluation script provided:

```bash
python evaluate.py --model_checkpoint rl_model --max_length 128
```

This script loads the model checkpoint saved in the `rl_model` directory and evaluates it on the XSTest dataset. The heuristic evaluator checks the model's responses based on:

- **Safe examples:** Ensuring the model provides a helpful answer without overrefusal.
- **Unsafe examples:** Verifying that the model produces an appropriate safe (refusal) completion.

The evaluation output includes overall accuracy, as well as separate accuracy metrics for safe and unsafe examples.

---

## Acknowledgements & Limitations

This repository is built upon cutting-edge research in alignment and safety. A gentle reminder: **the current implementation does not support multiprocessing yet** because the underlying unsloth multiprocessing feature is not available for free.

We hope you enjoy exploring this reimplementation and that it serves as a robust foundation for further research into safe and ethical AI alignment.

Happy Training!
