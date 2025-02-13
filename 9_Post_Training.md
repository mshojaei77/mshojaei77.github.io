---
title: "Post Training Techniques"
nav_order: 10
---

# Module 9: Post Training Techniques

![Post Training Techniques](image_url)

## Overview
This module covers essential post-training techniques for language models, including fine-tuning, parameter-efficient methods, model merging, and advanced techniques like GRPO.

## 1. Fine-tuning & Parameter-Efficient Techniques
Learn about various approaches to adapt pre-trained models efficiently.

### Core Materials
**Hands-on Practice:**
- **[Kaggle Gemma2 9b Unsloth notebook](https://kaggle.com/code/danielhanchen/kaggle-gemma2-9b-unsloth-notebook)**
- **[Quick Gemma-2B Fine-tuning Notebook](https://colab.research.google.com/drive/12OkGVWuh2lcrokExYhskSJeKrLzdmq4T?usp=sharing)**
- **[Phi-4 Finetuning Tutorial](https://www.kaggle.com/code/unsloth/phi-4-finetuning)**
- **[Fine-tuning Gemma 2 with LoRA](https://kaggle.com/code/iamleonie/fine-tuning-gemma-2-jpn-for-yomigana-with-lora)**

### Key Concepts
- Learning rate scheduling
- Batch size optimization 
- Gradient accumulation
- Early stopping
- Validation strategies
- Model checkpointing
- LoRA adapters
- QLoRA
- Prefix tuning
- Prompt tuning
- Adapter tuning
- BitFit
- IA3
- Soft prompts
- Parameter-efficient transfer learning

### Tools & Frameworks
**Core Tools:**
[![Hugging Face PEFT](https://badgen.net/badge/Framework/Hugging%20Face%20PEFT/green)](https://huggingface.co/docs/peft)
[![Lightning AI](https://badgen.net/badge/Framework/Lightning%20AI/green)](https://lightning.ai/)
[![Kaggle](https://badgen.net/badge/Website/Kaggle/blue)](https://www.kaggle.com/)

**Additional Tools:**
[![UnslothAI](https://badgen.net/badge/Github%20Repository/UnslothAI/cyan)](https://github.com/unslothai)
[![io.net](https://badgen.net/badge/API%20Provider/io.net/blue)](https://io.net/)

### Additional Resources
[![Fine-Tuning Transformers](https://badgen.net/badge/Docs/Fine-Tuning%20Transformers/green)](https://huggingface.co/docs/transformers/training)
[![How to Fine-Tune LLMs in 2024 with Hugging Face](https://badgen.net/badge/Blog/How%20to%20Fine-Tune%20LLMs%20in%202024%20with%20Hugging%20Face/pink)](https://philschmid.de/fine-tune-llms-in-2024-with-trl)
[![LoRA: Low-Rank Adaptation](https://badgen.net/badge/Blog/LoRA:%20Low-Rank%20Adaptation/pink)](https://huggingface.co/blog/lora)
[![Practical Tips for Finetuning LLMs Using LoRA](https://badgen.net/badge/Blog/Practical%20Tips%20for%20Finetuning%20LLMs%20Using%20LoRA/pink)](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
[![DataCamp Fine-tuning Tutorial](https://badgen.net/badge/Tutorial/DataCamp%20Fine-tuning%20Tutorial/blue)](https://www.datacamp.com/tutorial/fine-tuning-large-language-models)
[![ColPali Fine-tuning Tutorial](https://badgen.net/badge/Tutorial/ColPali%20Fine-tuning%20Tutorial/blue)](https://github.com/merveenoyan/smol-vision/blob/main/Finetune_ColPali.ipynb)
[![Parameter Freezing Strategies](https://badgen.net/badge/Paper/Parameter%20Freezing%20Strategies/purple)](https://arxiv.org/abs/2501.07818)
[![Memory-Efficient LoRA-FA](https://badgen.net/badge/Paper/Memory-Efficient%20LoRA-FA/purple)](https://arxiv.org/abs/2308.03303)

## 2. Advanced Fine-tuning Techniques
Understanding advanced optimization and alignment techniques.

### Core Materials
**Hands-on Practice:**
- **[Knowledge Distillation Basics](notebooks/knowledge_distillation_basics.ipynb)**
- **[Student Model Training](notebooks/student_model_training.ipynb)**

### Key Concepts
- Direct preference optimization
- Proximal policy optimization
- Constitutional AI
- Reward modeling
- Human feedback integration
- Curriculum learning

### Tools & Frameworks
[![TRL (Transformer Reinforcement Learning)](https://badgen.net/badge/Github%20Repository/TRL%20(Transformer%20Reinforcement%20Learning)/cyan)](https://github.com/huggingface/trl)

### Additional Resources
[![How to align open LLMs in 2025 with DPO & synthetic data](https://badgen.net/badge/Blog/How%20to%20align%20open%20LLMs%20in%202025%20with%20DPO%20%26%20synthetic%20data/pink)](https://philschmid.de/rl-with-llms-in-2025-dpo)
[![How to Fine-Tune LLMs in 2024 with Hugging Face](https://badgen.net/badge/Blog/How%20to%20Fine-Tune%20LLMs%20in%202024%20with%20Hugging%20Face/pink)](https://philschmid.de/fine-tune-llms-in-2024-with-trl)
[![Multi-Task Fine-tuning](https://badgen.net/badge/Paper/Multi-Task%20Fine-tuning/purple)](https://arxiv.org/abs/2408.03094)
[![Few-Shot Learning Approaches](https://badgen.net/badge/Paper/Few-Shot%20Learning%20Approaches/purple)](https://arxiv.org/html/2408.13296v1)

## 3. Model Merging
Learn to combine multiple fine-tuned models and merge model weights to create enhanced capabilities.

### Core Materials
**Hands-on Practice:**
- **[Model Merging Basics](notebooks/model_merging_basics.ipynb)**
- **[Weight Averaging](notebooks/weight_averaging.ipynb)**
- **[Model Fusion](notebooks/model_fusion.ipynb)**

### Key Concepts
- Weight averaging
- Model fusion 
- Task composition
- Knowledge distillation
- Parameter merging
- Model ensembling

### Tools & Frameworks
**Core Tools:**
[![mergekit](https://badgen.net/badge/Github%20Repository/mergekit/cyan)](https://github.com/cg123/mergekit)
[![LM-Model-Merger](https://badgen.net/badge/Github%20Repository/LM-Model-Merger/cyan)](https://github.com/lm-sys/LM-Model-Merger)

**Additional Tools:**
[![SLERP](https://badgen.net/badge/Github%20Repository/SLERP/cyan)](https://github.com/johnsmith0031/slerp_pytorch)
[![HuggingFace Model Merging Tools](https://badgen.net/badge/Hugging%20Face%20Space/HuggingFace%20Model%20Merging%20Tools/yellow)](https://huggingface.co/spaces/huggingface-projects/Model-Merger)

### Additional Resources
**Essential Resources:**
[![Merging Language Models](https://badgen.net/badge/Paper/Merging%20Language%20Models/purple)](https://arxiv.org/abs/2401.10597)
[![Weight Averaging Guide](https://badgen.net/badge/Blog/Weight%20Averaging%20Guide/pink)](https://huggingface.co/blog/merge-models)

**Additional Resources:**
[![Task Arithmetic with Language Models](https://badgen.net/badge/Paper/Task%20Arithmetic%20with%20Language%20Models/purple)](https://arxiv.org/abs/2212.04089)
[![Parameter-Efficient Model Fusion](https://badgen.net/badge/Paper/Parameter-Efficient%20Model%20Fusion/purple)](https://arxiv.org/abs/2310.13013)

## 4. Fine-tuning Datasets
Learn about curated datasets for instruction tuning, alignment, and specialized task adaptation of language models.

### Key Concepts
- Instruction tuning
- RLHF
- Task-specific data
- Data quality
- Prompt engineering
- Human feedback

### Tools & Frameworks
**Core Tools:**
[![Self-Instruct](https://badgen.net/badge/Github%20Repository/Self-Instruct/cyan)](https://github.com/yizhongw/self-instruct)
[![Argilla](https://badgen.net/badge/Github%20Repository/Argilla/cyan)](https://github.com/argilla-io/argilla)

**Additional Tools:**
[![LIDA](https://badgen.net/badge/Github%20Repository/LIDA/cyan)](https://github.com/microsoft/LIDA)
[![Stanford Alpaca Tools](https://badgen.net/badge/Github%20Repository/Stanford%20Alpaca%20Tools/cyan)](https://github.com/tatsu-lab/stanford_alpaca)

### Additional Resources
**Essential Resources:**
[![Anthropic's Constitutional AI](https://badgen.net/badge/Website/Anthropic's%20Constitutional%20AI/blue)](https://www.anthropic.com/research/constitutional)
[![OpenAI's InstructGPT Paper](https://badgen.net/badge/Paper/OpenAI's%20InstructGPT%20Paper/purple)](https://arxiv.org/abs/2203.02155)
[![DeepSeek-R1 Local Fine-tuning Guide](https://badgen.net/badge/Blog/DeepSeek-R1%20Local%20Fine-tuning%20Guide/pink)](https://x.com/_avichawla/status/1884126766132011149)

**Additional Resources:**
[![Self-Instruct Paper](https://badgen.net/badge/Paper/Self-Instruct%20Paper/purple)](https://arxiv.org/abs/2212.10560)
[![UltraFeedback Paper](https://badgen.net/badge/Paper/UltraFeedback%20Paper/purple)](https://arxiv.org/abs/2310.01377)

### Popular Datasets
| Dataset | Description |
|----------|-------------|
| [![Anthropic Constitutional AI Dataset](https://badgen.net/badge/Hugging%20Face%20Dataset/Anthropic%20Constitutional%20AI%20Dataset/yellow)](https://huggingface.co/datasets/anthropic/constitutional-ai) | Aligned instruction dataset with safety considerations |
| [![OpenAssistant Conversations](https://badgen.net/badge/Hugging%20Face%20Dataset/OpenAssistant%20Conversations/yellow)](https://huggingface.co/datasets/OpenAssistant/oasst1) | High-quality conversational data |
| [![UltraChat](https://badgen.net/badge/Hugging%20Face%20Dataset/UltraChat/yellow)](https://huggingface.co/datasets/HuggingFaceH4/ultrachat) | Large-scale chat interactions |
| [![UltraFeedback](https://badgen.net/badge/Hugging%20Face%20Dataset/UltraFeedback/yellow)](https://huggingface.co/datasets/openbmb/UltraFeedback) | Comprehensive model evaluation data |
| [![Synthia-Coder-v1.5-I](https://badgen.net/badge/Hugging%20Face%20Dataset/Synthia-Coder-v1.5-I/yellow)](https://huggingface.co/datasets/migtissera/Synthia-Coder-v1.5-I) | 23.5K coding samples from Claude Opus |
| [![Synthetic Medical Conversations](https://badgen.net/badge/Hugging%20Face%20Dataset/Synthetic%20Medical%20Conversations/yellow)](https://huggingface.co/datasets/OnDeviceMedNotes/synthetic-medical-conversations-deepseek-v3) | Multilingual medical dialogues |

## 5. Knowledge Distillation
Learn how to transfer expertise from large teacher models to smaller, efficient student models.

### Core Materials
**Practice Materials:**
- **Basic:** [Knowledge Distillation Basics](notebooks/knowledge_distillation_basics.ipynb)
- **Advanced:** [Student Model Training](notebooks/student_model_training.ipynb)

### Key Concepts
- Knowledge distillation
- Teacher-student architecture
- Model compression

### Tools & Frameworks
[![Hugging Face Transformers](https://badgen.net/badge/Framework/Hugging%20Face%20Transformers/green)](https://huggingface.co/docs/transformers)

### Additional Resources
**Essential Resources:**
[![Knowledge Distillation Explained](https://badgen.net/badge/Tutorial/Knowledge%20Distillation%20Explained/blue)](https://towardsdatascience.com/knowledge-distillation-simplified-ddc070724770)

**Additional Resources:**
[![DistilBERT Paper](https://badgen.net/badge/Paper/DistilBERT%20Paper/purple)](https://arxiv.org/abs/1910.01108)

## 6. Reasoning Models and GRPO
Learn about models that enhance reasoning capabilities through chain-of-thought and GRPO-based training.

### Core Materials
**Practice Materials:**
- **Basic:** [GRPO Poetry Generation](https://colab.research.google.com/drive/1Ty0ovsrpw8i-zJvDhlSAtBIVw3EZfHK5?usp=sharing)
- **Intermediate:** [Qwen 0.5B GRPO Implementation](https://colab.research.google.com/drive/1bfhs1FMLW3FGa8ydvkOZyBNxLYOu0Hev?usp=sharing)
- **Advanced:** [Phi-4 14B GRPO Training](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4_(14B)-GRPO.ipynb)
- **Expert:** [Llama 3.1 8B GRPO](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)

### Key Concepts
- Chain-of-thought
- Reasoning
- GRPO
- Preference learning
- Reward modeling
- Group-based advantage estimation
- Resource-efficient training
- Reasoning enhancement
- Reinforcement learning
- Long context scaling

### Tools & Frameworks
**Core Tools:**
[![Unsloth](https://badgen.net/badge/Github%20Repository/Unsloth/cyan)](https://github.com/unslothai/unsloth)
[![DeepSeek-R1 Training Framework](https://badgen.net/badge/Github%20Repository/DeepSeek-R1%20Training%20Framework/cyan)](https://github.com/deepseek-ai/DeepSeek-R1)

**Additional Tools:**
[![TinyZero](https://badgen.net/badge/Github%20Repository/TinyZero/cyan)](https://github.com/Jiayi-Pan/TinyZero)
[![Kimi.ai](https://badgen.net/badge/Website/Kimi.ai/blue)](https://kimi.ai)

### Additional Resources
**Essential Resources:**
[![DeepSeek R1 Reasoning Primer](https://badgen.net/badge/Blog/DeepSeek%20R1%20Reasoning%20Primer/pink)](https://aman.ai/primers/ai/deepseek-R1/)
[![DeepSeek GRPO Paper](https://badgen.net/badge/Paper/DeepSeek%20GRPO%20Paper/purple)](https://arxiv.org/pdf/2402.03300)
[![DeepSeek R1 Reasoning Blog](https://badgen.net/badge/Blog/DeepSeek%20R1%20Reasoning%20Blog/pink)](https://unsloth.ai/blog/r1-reasoning)
[![Kimi k1.5 Paper](https://badgen.net/badge/Paper/Kimi%20k1.5%20Paper/purple)](https://arxiv.org/abs/2401.12863)

**Additional Resources:**
[![DeepSeek R1 Theory Overview](https://badgen.net/badge/Video/DeepSeek%20R1%20Theory%20Overview/red)](https://www.youtube.com/watch?v=QdEuh2UVbu0)
[![GRPO Explained by Yannic Kilcher](https://badgen.net/badge/Video/GRPO%20Explained%20by%20Yannic%20Kilcher/red)](https://youtube.com/watch?v=bAWV_yrqx4w)
[![Open-R1](https://badgen.net/badge/Blog/Open-R1/pink)](https://huggingface.co/blog/open-r1/update-1)
[![AGIEntry Kimi Overview](https://badgen.net/badge/Website/AGIEntry%20Kimi%20Overview/blue)](https://agientry.com)

### GRPO Datasets
| Dataset | Description |
|----------|-------------|
| [![PleIAs Verse Wikisource](https://badgen.net/badge/Hugging%20Face%20Dataset/PleIAs%20Verse%20Wikisource/yellow)](https://huggingface.co/datasets/PleIAs/verse-wikisource) | 200,000 verses for poetry training |
| [![Bespoke-Stratos-17k](https://badgen.net/badge/Hugging%20Face%20Dataset/Bespoke-Stratos-17k/yellow)](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) | High-quality CoT with generation code |
| [![OpenThoughts-114k](https://badgen.net/badge/Hugging%20Face%20Dataset/OpenThoughts-114k/yellow)](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) | Comprehensive reasoning patterns distilled from R1 |
| [![Evalchemy Dataset](https://badgen.net/badge/Hugging%20Face%20Dataset/Evalchemy%20Dataset/yellow)](https://huggingface.co/datasets/evalchemy) | Complementary reasoning dataset |
| [![R1-Distill-SFT](https://badgen.net/badge/Hugging%20Face%20Dataset/R1-Distill-SFT/yellow)](https://huggingface.co/datasets/ServiceNow-AI/R1-Distill-SFT) | 1.8M samples from DeepSeek-R1-32b |
| [![Sky-T1_data_17k](https://badgen.net/badge/Hugging%20Face%20Dataset/Sky-T1_data_17k/yellow)](https://huggingface.co/datasets/NovaSky-AI/Sky-T1_data_17k) | 17k verified samples for coding, math, science |
| [![SCP-116K](https://badgen.net/badge/Hugging%20Face%20Dataset/SCP-116K/yellow)](https://huggingface.co/datasets/EricLu/SCP-116K) | Scientific problem-solving (Physics, Chemistry, Biology) |
| [![FineQwQ-142k](https://badgen.net/badge/Hugging%20Face%20Dataset/FineQwQ-142k/yellow)](https://huggingface.co/datasets/qingy2024/FineQwQ-142k) | Math, Coding, General reasoning |
| [![Dolphin-R1](https://badgen.net/badge/Hugging%20Face%20Dataset/Dolphin-R1/yellow)](https://huggingface.co/datasets/cognitivecomputations/dolphin-r1) | Combined R1 and Gemini 2 reasoning |
| [![Dolphin-R1-DeepSeek](https://badgen.net/badge/Hugging%20Face%20Dataset/Dolphin-R1-DeepSeek/yellow)](https://huggingface.co/datasets/mlabonne/dolphin-r1-deepseek) | DeepSeek-compatible format |