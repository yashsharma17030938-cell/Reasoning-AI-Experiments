# 🧠 Reasoning-AI-Experiments

> **A one-day engineering project to build, train, and test custom Reinforcement Learning (RL) pipelines for Large Language Models using Group Relative Policy Optimization (GRPO).**

## 📖 Overview
This repository documents the journey of taking standard, next-token prediction language models and forcing them to develop **System 2 thinking** (slow, deliberate, step-by-step reasoning). 

Instead of traditional Supervised Fine-Tuning (SFT), these experiments utilize Reinforcement Learning to teach models *how* to think, using mathematical ground-truth as an absolute reward filter.

---

## 🧪 Experiment 01: The 1.5B Proof-of-Concept
The goal of this initial 24-hour project was to verify the training pipeline, handle extreme hardware constraints (Google Colab 16GB T4 GPU), and establish a working Process Reward environment.

### The Stack
* **Base Model:** `Qwen2.5-1.5B-Instruct`
* **Optimization:** 4-bit Quantization via `Unsloth` & LoRA (Low-Rank Adaptation)
* **Algorithm:** GRPO via Hugging Face `TRL`
* **Environment:** GSM8K Dataset (Grade School Math)

### The Reward Function (The Dopamine Loop)
The model was mathematically forced to output its internal monologue using XML tags (`<reasoning>` and `<answer>`). It was graded strictly by a Python compilation script:
1. **Format Reward (+0.5):** Did the model successfully isolate its thoughts from its final answer?
2. **Correctness Reward (+1.0):** Did the model's final extracted string match the exact mathematical ground-truth?

### 📈 Results & Observations
The model was run for a 300-step micro-epoch. 
* **Baseline:** Prior to training, the 1.5B model failed to structure its thoughts and frequently hallucinated math, resulting in a baseline reward of `0.000`.
* **Post-Training:** By step 50, the RL algorithm caught a successful logic branch. By step 300, the model was consistently achieving batch rewards of `0.500` to `0.750`.
* **Cognitive Shift:** While a 1.5B model lacks the parameter count for advanced competitive reasoning, the structural goal was achieved. The model learned *patience*. It physically updated its weights to prioritize writing long-form reasoning before attempting a final answer, proving the GRPO pipeline functions perfectly.

---

## 🚀 Future Roadmap: Llama 3 & Beyond
With the core architecture validated, the next phases of this project involve:
1. **Model Upgrade:** Swapping the 1.5B dummy model for `Meta-Llama-3.1-8B-Instruct`.
2. **Dataset Upgrade:** Moving from GSM8K to the Hendrycks `MATH` dataset (competition-level Olympiad problems).
3. **Advanced Reward Functions:** Implementing custom Python scripts to programmatically strip and verify complex LaTeX algebraic outputs.

## ⚙️ How to Run
*(Note: Scripts require a minimum of 15GB VRAM and heavily utilize Unsloth for memory management).*
1. Clone this repository.
2. Open the `.ipynb` files in Google Colab or Kaggle.
3. Set runtime to T4 GPU.
4. Execute the training loop (Ensure Google Drive is mounted for checkpoint auto-saving).
