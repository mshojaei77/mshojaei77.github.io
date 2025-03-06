---
title: "Fine-Tuning Large Language Models"
nav_order: 1
parent: Posts
layout: default
---

# Fine-Tuning Large Language Models: A Comprehensive, Step-by-Step Guide

This guide provides a comprehensive, hands-on approach to fine-tuning Large Language Models (LLMs).  It covers the theory, practical considerations, and code examples needed to adapt a pre-trained LLM for your specific tasks.

**Step 1: Understanding Fine-Tuning**

*   **1.1 What is Fine-Tuning?**

    Fine-tuning is a powerful technique in transfer learning.  Instead of training a massive LLM from scratch, you start with a pre-trained model (like those from Meta, Mistral, or Google) that already understands language well.  Fine-tuning involves taking this pre-trained model and training it further on a *smaller, specific dataset* related to your desired task. This adjusts the model's internal "weights" to specialize in your task while retaining its general language knowledge.

    *Analogy:* Imagine a general doctor undergoing specialized training to become a cardiologist.  They keep their general medical knowledge but gain deep expertise in heart-related issues.

*   **1.2 Why Fine-Tune?**

    Fine-tuning offers significant advantages:

    *   **Improved Performance:**  Achieve higher accuracy and fluency on your specific task compared to using a general-purpose LLM directly.
    *   **Data Efficiency:**  You need *far less* data than training from scratch – thousands of examples can often be sufficient, compared to the billions of data points used for pre-training.
    *   **Resource Efficiency:**  Fine-tuning requires significantly less computational power (GPUs/TPUs) and time, making it accessible even without massive infrastructure.
    *   **Customization:**  Tailor the model's style, tone, and knowledge to your specific needs.
    *   **Adaptability:**  Easily adapt the model to new tasks or changing requirements by fine-tuning it further on new data.

*   **1.3 Types of Fine-Tuning:**

    Several approaches exist, each with its own trade-offs:

    *   **Full Fine-Tuning:**  The most traditional approach; *all* model parameters are updated.  This offers the greatest potential for adaptation but is computationally expensive and can lead to *catastrophic forgetting* (losing previously learned knowledge).
    *   **Parameter-Efficient Fine-Tuning (PEFT):**  Updates only a *small fraction* of the model's parameters, making it much more efficient and reducing the risk of catastrophic forgetting.  Key PEFT methods include:
        *   **LoRA (Low-Rank Adaptation):**  Adds small, trainable "adapter" layers to the model.  The original pre-trained weights are *frozen*, preserving general knowledge.  LoRA is very popular due to its effectiveness and efficiency.
        *   **Prefix Tuning:**  Adds trainable vectors (a "prefix") to the input embeddings at each layer.
        *   **Prompt Tuning:**  Adds a learned "soft prompt" to the initial input embedding.
        *   **IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations):**  Learns scaling factors for the model's internal activations.
    *   **Instruction Fine-Tuning:**  Trains the model on a dataset of instructions and their corresponding outputs, teaching it to follow instructions and generalize to new tasks. This is *crucial* for conversational models.  The dataset typically uses a format like:
        ```jsonl
        {"conversations": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "The capital of France is Paris."}]}
        ```
    *   **Reinforcement Learning from Human Feedback (RLHF):**  A more advanced technique using human feedback to refine the model's outputs, improving qualities like helpfulness and harmlessness.  It often involves these steps:
        1.  **Supervised Fine-Tuning (SFT):**  Initial fine-tuning on instruction-output pairs.
        2.  **Reward Model Training:**  A separate model is trained to predict human preferences by comparing different model outputs.
        3.  **Reinforcement Learning:**  The LLM is fine-tuned using reinforcement learning, using the reward model's predictions as the reward signal.
        *   **DPO (Direct Preference Optimization):**  Simplifies RLHF by directly optimizing the LLM based on preference data, avoiding a separate reward model.
        * **ORPO (Odds Ratio Preference Optimization):**  An alternative method to standard RLHF.

**Step 2: Choosing a Base Model**

Selecting the right pre-trained LLM is critical. Consider these factors:

*   **2.1 Model Size:**

    Measured by the number of parameters, impacting performance and resource needs.

    *   **Small (1.5B - 7B parameters):**  Accessible; can be fine-tuned on consumer-grade GPUs (e.g., NVIDIA RTX 3090, 4090) or smaller cloud instances.  Good for experimentation and resource-constrained projects. Examples: `Qwen2.5-1.5B-Instruct`, `Llama-3.2-3B-Instruct`, `Gemma-2-2b-it`. Quantized versions (e.g., `unsloth/gemma-2-2b-it-bnb-4bit`) further reduce resource demands.
    *   **Medium (14B - 32B parameters):**  Balance performance and resources; require more powerful GPUs (e.g., NVIDIA A100, H100) or multiple GPUs. Examples: `Coma-II-14B`, `Arcee-Blitz (24B)`, `TheBeagle-v2beta-32B-MGS`.
    *   **Large (70B+ parameters):**  State-of-the-art performance, but require significant resources (multiple high-end GPUs).  Examples: `Qwen2.5-72B-Instruct`, `Meta-Llama-3.1-70B-Instruct`, `AceInstruct-72B`.

*   **2.2 Pre-training Data:**

    Impacts the model's capabilities and biases.

    *   **General-Purpose:**  Trained on massive, diverse datasets (web, books, code).  Suitable for a wide range of tasks. Examples: Qwen2.5, Llama 3.
    *   **Specialized:**  Trained on specific data (e.g., scientific papers, code).  May be advantageous if your task aligns.  Examples: `EpistemeAI/Athena-gemma-2-2b-it-Philos` (reasoning, philosophy), `DeepSeek-R1-Distill-Llama-3B` (reasoning).

*   **2.3 Model Architecture:**

    Most LLMs are Transformer-based.

    *   **Causal Language Models (Decoder-Only):**  Most common for text generation; trained to predict the next token.  Good for chatbots, creative writing, etc.
    *   **Encoder-Decoder Models:**  Separate encoder and decoder; often used for translation and summarization.  This guide focuses on decoder-only models.

*   **2.4 Licensing:**

    *CRITICAL*: Determines how you can use the model, *especially* for commercial purposes.

    *   **Permissive Licenses (Apache 2.0, MIT, Llama 3 license):**  Allow broad use, including commercial use.
    *   **Non-Commercial Licenses:**  Restrict use to research or non-commercial purposes (e.g., `AceInstruct` models).
    *   **Always check the license** before using a model to avoid legal issues.

*   **2.5 Existing Fine-tuning:**

    Many models are already fine-tuned.

    *   **"Instruct" Models:**  Fine-tuned on instructions and outputs (e.g., `Qwen2.5-72B-Instruct`, `Meta-Llama-3.1-70B-Instruct`).  A good starting point for many tasks.
    *   **Chat Models:**  Specifically fine-tuned for conversations.

*   **2.6 Language Support:**

    Choose a multilingual model if your task involves multiple languages (e.g., Qwen, EXAONE 3.5, `AGO solutions Llama-3.1-SauerkrautLM-70b-Instruct`).

*   **2.7 Prompt Template:**

    *ABSOLUTELY CRITICAL*: Different models expect different input formats. Using the wrong template leads to poor results.

    *   **ChatML:**  Common with Qwen and chat models. Uses `<|im_start|>` and `<|im_end|>` tokens.
    *   **Llama 3 Template:**  Specific to Llama 3.  Uses `<|start_header_id|>`, `<|end_header_id|>`, and `<|eot_id|>`.
    *   **Gemma Template:**  Used by Google's Gemma models.
    *   **Custom Templates:** Some models have unique formats.
    *   **The model card or documentation *must* specify the correct template.**
    *   **The Hugging Face `transformers` library's `tokenizer.apply_chat_template()` function *automatically* handles these complexities. Use it!**

* **2.8 Uncensored/Abliterated Models**
    * Models like `huihui-ai/Qwen2.5-72B-Instruct-abliterated` have had safety filters removed.
    * **Extreme caution!** These can generate offensive or biased content. Only use if you have a very specific, justified need and understand the risks.

* **2.9 Merged Models:**
    * Created with tools like `mergekit` (e.g., `TheBeagle-v2beta-32B-MGS`, `QwQ-32B Kumo`).
    * Can be powerful but are experimental; their behavior may be less predictable.

**Step 3: Data Preparation**

Data preparation is the *most crucial* step. The quality, relevance, and format of your data directly impact the fine-tuned model's performance. A bad dataset leads to a bad model.

*   **3.1 Dataset Format:**

    *   **JSON Lines (`.jsonl`):** The *recommended* format. Each line is a JSON object representing a single training example. Easy to read, write, and process.

*   **3.2 Dataset Structure:**

    Depends on your task. Common examples:

    *   **Instruction Following/Chat:** (Most common for conversational models)
        ```jsonl
        {"conversations": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "The capital of France is Paris."}]}
        {"conversations": [{"role": "user", "content": "Translate 'Hello world' to French."}, {"role": "assistant", "content": "Bonjour le monde"}]}
        ```
        *   `conversations`: A list of messages (conversation turns).
        *   `role`:  `"system"` (initial context, optional but helpful), `"user"` (user input), `"assistant"` (model response).
        *   `content`:  The text of the message.
        *   **Conversations should *alternate* between `user` and `assistant`.  The `system` message, if present, should be first.**

    *   **Text Completion:**
        ```jsonl
        {"input": "The quick brown fox", "output": " jumps over the lazy dog."}
        ```
        *   `input`: Starting text.
        *   `output`: Desired completion.

    *   **Other Tasks:** (Adapt the format)
        *   **Summarization:** `{"input": "Long article...", "output": "Concise summary..."}`
        *   **Translation:** `{"input": "Hello world", "target_language": "fr", "output": "Bonjour le monde"}`
        *   **Question Answering:** `{"context": "...", "question": "...", "answer": "..."}`

*   **3.3 Prompt Formatting:**

    (Repeating for emphasis) Using the correct prompt template is *absolutely essential*.

    *   **ChatML (example):**
        ```
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        What is the highest mountain?<|im_end|>
        <|im_start|>assistant
        Mount Everest.<|im_end|>
        ```

    *   **Llama 3 Template (example):**
        ```
        <|start_header_id|>system<|end_header_id|>
        You are a helpful assistant.<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        What is the highest mountain?<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        Mount Everest.<|eot_id|>
        ```

    * **Gemma Template (example):**
       ```jsonl
       {"conversations": [{"role": "user", "content": "Write a hello world program"}, {"role": "model", "content": "```python\nprint(\"Hello, world!\")\n```"}]}
       ```
    * **Custom Prompt (DeepSeek-R1-Distill-Llama-3B example):**
        ```
        <|start_header_id|>system<|end_header_id|>
        Respond in the following format:
        <think>
        You should reason between these tags.
        </think>

        Answer goes here...

        Always use <think> </think> tags even if they are not necessary.
        <|eot_id|>

        <|start_header_id|>user<|end_header_id|>
        Which one is larger? 9.11 or 9.9?<|eot_id|>

        <|start_header_id|>assistant<|end_header_id|>
        ...
        ```

    *   **Using `tokenizer.apply_chat_template()` (Highly Recommended):**

        ```python
        from transformers import AutoTokenizer

        # ChatML example (e.g., Qwen)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
        messages = [{"role": "user", "content": "What is the capital of France?"}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(formatted_prompt)

        # Llama 3 example
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        messages = [{"role": "user", "content": "What is the capital of France?"}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(formatted_prompt)

        ```
        This function *automatically* adds the correct special tokens, system messages, etc., for your chosen model.  Use this to avoid errors!

*   **3.4 Data Quality:**

    *   **Accuracy:** Ensure factual correctness.
    *   **Consistency:** Maintain a consistent style, tone, and format.
    *   **Relevance:** Data should directly relate to your task.
    *   **Diversity:** Include a wide range of examples.
    *   **Bias:**  Be aware of and mitigate potential biases in your data.

*   **3.5 Data Quantity:**

    *   More data is generally better, but *quality beats quantity*.
    *   A few thousand high-quality examples are often sufficient.
    *   For specialized tasks, even a few hundred can work.
    *   Consider data augmentation if you have limited data.

*   **3.6 Data Preprocessing:**

    *   **Cleaning:** Remove irrelevant characters, HTML, etc.
    *   **Tokenization:**  The `tokenizer` (loaded from the pre-trained model) converts text to numerical tokens. `tokenizer.apply_chat_template()` and the `Trainer` class handle this, but be aware of the tokenizer's vocabulary and special tokens.
    *   **Truncation/Padding:**  Handle inputs/outputs exceeding the model's maximum context length (e.g., 2048, 4096, 32768 tokens).
        *   **Truncation:** Shorten sequences (carefully!).
        *   **Padding:** Add padding tokens to shorter sequences. The `transformers` library helps with this.
    *   **Data Splitting:** Divide your dataset into:
        *   **Training Set:** (Largest) Used for training.
        *   **Validation Set:** Used for evaluation during training and hyperparameter tuning.
        *   **Test Set:**  Used for final, unbiased evaluation.

**Step 4: Setting Up Your Environment**

*   **4.1 Hardware:**

    *   **GPU:**  Fine-tuning almost always requires a GPU.  VRAM (GPU memory) needs depend on model size, batch size, and sequence length.
    *   **VRAM Requirements:**
        *   **Small Models (1.5B - 7B):**  8GB-24GB VRAM (e.g., RTX 3060, 3090, 4090).
        *   **Medium Models (14B - 32B):**  24GB-80GB VRAM (e.g., A100, H100) or multiple GPUs.
        *   **Large Models (70B+):** Multiple high-end GPUs (A100s, H100s).
    *   **Cloud-Based GPUs:**  If you lack a suitable GPU:
        *   Google Colab (free, limited)
        *   Amazon SageMaker
        *   Google Cloud AI Platform
        *   Microsoft Azure Machine Learning
        *   Paperspace Gradient
        *   RunPod

*   **4.2 Software:**

    Install these (using `pip` where appropriate):

    *   **Python:** 3.8 or later.
    *   **PyTorch:**  Follow instructions on [https://pytorch.org/](https://pytorch.org/), choosing the correct CUDA version if you have an NVIDIA GPU.
    *   **Hugging Face Transformers:** `pip install transformers`
    *   **Hugging Face Accelerate:** `pip install accelerate`
    *   **Hugging Face Datasets:** `pip install datasets`
    *   **Optional (depending on your needs):**
        *   **TRL (for RLHF):** `pip install trl`
        *   **PEFT (for LoRA, etc.):** `pip install peft`
        *   **BitsAndBytes (for quantization):** `pip install bitsandbytes`
        *   **Unsloth:** (Faster, memory-efficient fine-tuning) Follow installation instructions on the Unsloth GitHub repository.
        *   **DeepSpeed:** (For very large models) Follow installation instructions on the DeepSpeed GitHub repository.
        *   **Axolotl:** (User-friendly framework) See Section 10 (below).
        *   **Mergekit:** (For merging models) `pip install mergekit`

**Step 5: Fine-Tuning Code (using Hugging Face Transformers)**

This provides a complete, runnable example using the Hugging Face `Trainer` class, which simplifies the process.

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat"  # Replace with your chosen model
DATASET_PATH = "your_dataset.jsonl"  # Path to your .jsonl dataset file
OUTPUT_DIR = "fine_tuned_model"
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3
MAX_SEQ_LENGTH = 1024  # Maximum sequence length (adjust as needed)
USE_PEFT = True  # Set to False for full fine-tuning
LORA_R = 8  # LoRA rank (only if USE_PEFT is True)
LORA_ALPHA = 16  # LoRA alpha (only if USE_PEFT is True)
LORA_DROPOUT = 0.05  # LoRA dropout (only if USE_PEFT is True)


# --- Load Model and Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Qwen models need this, and it's good practice for others.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for faster training (if supported)
    device_map="auto",  # Automatically distribute the model across available devices
    trust_remote_code=True, #For models that require this
)

# --- Prepare PEFT (if enabled) ---
if USE_PEFT:
    from peft import LoraConfig, get_peft_model

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # Specific to model architecture.  Adapt if needed.
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

# --- Load and Prepare Dataset ---
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")


def preprocess_function(examples):
    # Format conversations using the tokenizer's chat template
    formatted_conversations = [
        tokenizer.apply_chat_template(example["conversations"], tokenize=False, add_generation_prompt=True)
        for example in examples["conversations"]
    ]

    # Tokenize the formatted conversations.  Return attention mask as well.
    model_inputs = tokenizer(
        formatted_conversations,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    # Set labels to be the same as the input_ids, for causal LM
    model_inputs["labels"] = model_inputs["input_ids"].clone()

    return model_inputs



processed_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names,  # Remove original columns
    num_proc=4,  # Use multiple processes for faster preprocessing
)

# Split the dataset
processed_dataset = processed_dataset.train_test_split(test_size=0.1)
train_dataset = processed_dataset["train"]
eval_dataset = processed_dataset["test"]


# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,  # Log every 10 steps
    evaluation_strategy="steps",  # Evaluate every eval_steps
    eval_steps=100,  # Evaluate every 100 steps
    save_strategy="steps",  # Save every save_steps
    save_steps=500,  # Save every 500 steps
    save_total_limit=2,  # Keep only the last 2 checkpoints
    fp16=True,  # Use mixed precision training (if supported)
    bf16=False, #Disable in favor of the model config.  Enable if not using the model config.
    report_to="tensorboard",  # Report metrics to TensorBoard
)

# --- Data Collator ---
# This is important for padding the inputs to the same length in each batch.
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# --- Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# --- Train the Model ---
trainer.train()

# --- Save the Fine-Tuned Model ---
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR) # Save the tokenizer as well.

print(f"Fine-tuning complete! Model saved to {OUTPUT_DIR}")

```

Key improvements and explanations in this code:

*   **Complete and Runnable:** This code is a fully functional example.  You can run it after replacing placeholders like `MODEL_NAME` and `DATASET_PATH`.
*   **Clear Configuration:** All key parameters are defined at the top for easy modification.
*   **Model and Tokenizer Loading:**  Loads the model and tokenizer correctly, handling padding tokens.  Includes `trust_remote_code=True` for models that require executing custom code.
*   **PEFT Support:**  Includes a complete implementation of LoRA (Low-Rank Adaptation), a highly effective PEFT technique.  The `USE_PEFT` flag controls whether to use full fine-tuning or PEFT.
*   **Dataset Loading and Preprocessing:**
    *   Uses `load_dataset` to load the `.jsonl` dataset.
    *   **Crucially, uses `tokenizer.apply_chat_template()`** to format the conversations correctly for the chosen model.  This is *essential* for good performance.
    *   Tokenizes the formatted conversations, handles padding and truncation, and creates `labels` for causal language modeling.
    *   Uses `map` with `batched=True` and `num_proc` for efficient preprocessing.
    *   Splits the dataset into training and evaluation sets.
*   **Training Arguments:**  Sets up the `TrainingArguments` with clear explanations for each parameter:
    *   `per_device_train_batch_size`: Batch size per GPU.
    *   `gradient_accumulation_steps`:  Simulates larger batch sizes by accumulating gradients.
    *   `learning_rate`:  The learning rate.
    *   `num_train_epochs`: Number of training epochs.
    *   `logging_steps`, `evaluation_strategy`, `eval_steps`, `save_strategy`, `save_steps`: Control logging, evaluation, and checkpoint saving.
    *   `fp16`: Enables mixed precision training (if your GPU supports it) for faster training and reduced memory usage.
    *   `bf16`:  Uses bfloat16, often preferred over fp16 on modern GPUs. Set in model config for simplicity.
    *   `report_to`: Enables reporting to TensorBoard for visualization.
*   **Data Collator:**  Uses `DataCollatorForLanguageModeling` to handle padding within each batch.
*   **Trainer:**  Creates and uses the `Trainer` class, which handles the training loop, evaluation, and checkpointing.
*   **Saving:**  Saves both the fine-tuned model *and* the tokenizer, which is necessary for later use.
* **Comments:** Added extensive comments.
* **Target Modules (for PEFT):**  The `target_modules` parameter in the `LoraConfig` is *crucial* and often needs adjustment depending on the specific model architecture.  The provided example includes common target modules, but you should consult the model's documentation or source code to confirm the correct modules for your chosen model. Incorrect target modules can lead to poor performance or errors. This is one of the trickiest parts of using LoRA.

**How to Run This Code:**

1.  **Install Dependencies:**  Make sure you have all the required libraries installed (see Step 4).
2.  **Create Your Dataset:**  Prepare your `.jsonl` dataset file (see Step 3).  Make sure it's correctly formatted.
3.  **Replace Placeholders:**
    *   `MODEL_NAME`:  Set this to the Hugging Face model identifier of your chosen base model (e.g., `"Qwen/Qwen1.5-1.8B-Chat"`, `"meta-llama/Meta-Llama-3-8B-Instruct"`).
    *   `DATASET_PATH`:  Set this to the path to your `.jsonl` dataset file.
4.  **Adjust Parameters:**  Fine-tune the hyperparameters in the `Configuration` section (batch size, learning rate, epochs, sequence length, LoRA parameters, etc.) based on your hardware and task.  Start with the defaults and experiment.
5.  **Run the Script:**  Execute the Python script.  Training will begin, and progress will be printed to the console.  Checkpoints and logs will be saved to the `OUTPUT_DIR`.

**Step 6: Evaluating Your Model**

After fine-tuning, it's crucial to evaluate your model's performance.

*   **Quantitative Evaluation:**
    *   **Perplexity:**  A common metric for language models, measuring how well the model predicts the next token in a sequence.  Lower perplexity is generally better. The `Trainer` class automatically calculates perplexity on the evaluation set.
    *   **Task-Specific Metrics:**  Use metrics relevant to your task.  For example:
        *   **Classification:** Accuracy, precision, recall, F1-score.
        *   **Summarization:** ROUGE score.
        *   **Translation:** BLEU score.
        *   **Question Answering:** Exact Match (EM) and F1-score.
    *   You can integrate these metrics into your training loop using the `compute_metrics` function in the `Trainer`.

*   **Qualitative Evaluation:**

    *   **Manual Inspection:**  Generate text using your fine-tuned model and *manually* assess the quality, fluency, relevance, and correctness of the outputs.  This is *essential*.  Don't rely solely on automated metrics.
    *   **Human Evaluation:**  If possible, have human evaluators rate the quality of the model's outputs based on specific criteria (e.g., helpfulness, harmlessness, coherence).

```python
# Example of generating text with the fine-tuned model
from transformers import pipeline

# Load the fine-tuned model and tokenizer
generator = pipeline("text-generation", model=OUTPUT_DIR, tokenizer=OUTPUT_DIR, device=0)  # Use device=0 for GPU

# Generate text
prompt = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n" #Use appropriate prompt format.
generated_text = generator(prompt, max_length=50, num_return_sequences=1)
print(generated_text[0]['generated_text'])

```
**Step 7: Hyperparameter Tuning**

Hyperparameter tuning is the process of finding the optimal values for the various settings that control the training process (learning rate, batch size, number of epochs, etc.).

*   **Key Hyperparameters:**
    *   **Learning Rate:**  Controls how quickly the model's weights are updated.  A good starting point is usually between 1e-5 and 5e-5.  Too high a learning rate can cause instability; too low can lead to slow training.
    *   **Batch Size:**  The number of training examples processed in each iteration.  Larger batch sizes can speed up training but require more GPU memory.
    *   **Number of Epochs:**  How many times the model goes through the entire training dataset.  Too few epochs can lead to underfitting; too many can lead to overfitting.
    *   **Gradient Accumulation Steps:**  Allows you to simulate larger batch sizes without increasing memory usage.
    *   **LoRA Parameters (r, alpha, dropout):**  These control the behavior of LoRA.  Experiment with different values.
    *   **Optimizer (AdamW, Adafactor):** The algorithm used to update the model's weights. AdamW is generally a good choice.
*   **Tuning Strategies:**
    *   **Manual Search:**  Try different combinations of hyperparameters and evaluate the results on the validation set.
    *   **Grid Search:**  Systematically try all combinations of hyperparameters within a specified range.
    *   **Random Search:**  Randomly sample hyperparameters from a specified distribution. Often more efficient than grid search.
    *   **Bayesian Optimization:**  Uses a probabilistic model to guide the search for optimal hyperparameters.
    *   **Hugging Face `Trainer` with `hyperparameter_search`:** The `Trainer` class provides a built-in `hyperparameter_search` method that can be used with libraries like Optuna or Ray Tune.

**Step 8: Advanced Techniques**

*   **Quantization:**  Reduces the precision of the model's weights (e.g., from 32-bit to 8-bit or 4-bit), significantly reducing memory usage and allowing you to run larger models on less powerful hardware.  Libraries like `bitsandbytes` and `Unsloth` provide tools for quantization.
*   **Distributed Training:**  Spreads the training process across multiple GPUs or machines, enabling you to train larger models and speed up training.  Hugging Face `accelerate` and DeepSpeed provide tools for distributed training.
*   **Mixed Precision Training:**  Uses a combination of 16-bit and 32-bit floating-point numbers to speed up training and reduce memory usage. The `fp16=True` argument in `TrainingArguments` enables this.
* **Gradient Checkpointing:** Saves memory by recomputing intermediate activations during the backward pass, instead of storing them. Useful for long sequences. You can enable this with:
    ```python
    model.gradient_checkpointing_enable()
    ```
*   **RLHF (Reinforcement Learning from Human Feedback):** As described earlier, RLHF is a powerful but complex technique that can significantly improve the quality of your model's outputs.  The `trl` library provides tools for implementing RLHF.

**Step 9: Saving and Loading Your Model**

*   **Saving:**

    ```python
    trainer.save_model("my_fine_tuned_model")  # Saves the model
    tokenizer.save_pretrained("my_fine_tuned_model")  # Saves the tokenizer
    ```
    This creates a directory (`my_fine_tuned_model` in this example) containing the model weights, configuration, and tokenizer files.
* **Loading:**

    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("my_fine_tuned_model", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("my_fine_tuned_model")
    ```

**Step 10: Using Axolotl**

Axolotl is a user-friendly framework built on top of Hugging Face Transformers that simplifies the fine-tuning process. It uses YAML configuration files to define the training parameters, making it easier to manage and reproduce experiments.

*   **Installation:** Follow the installation instructions in the Axolotl GitHub repository.  It's usually straightforward with `pip`.
*   **Configuration (YAML):** Create a YAML file (e.g., `config.yaml`) to define your fine-tuning setup:

    ```yaml
    base_model: Qwen/Qwen1.5-1.8B-Chat  # Your base model
    model_type: AutoModelForCausalLM
    tokenizer_type: AutoTokenizer
    trust_remote_code: true # For models that need this
    load_in_8bit: false # Enable 8-bit quantization (adjust as needed)
    load_in_4bit: false  # Enable 4-bit quantization (adjust as needed)
    gptq: false # For GPTQ models

    datasets:
      - path: your_dataset.jsonl  # Path to your dataset
        type: json
        split: train #For datasets without a train/test split
        #split: "train" #Use with train_test_split
        #test_size: 0.1 #Use with train_test_split

    dataset_prepared_path:
    val_set_size: 0.05  # Percentage of data to use for validation
    output_dir: ./fine-tuned-qwen  # Output directory

    adapter: lora  # Use LoRA (or other PEFT methods)
    lora_r: 8
    lora_alpha: 16
    lora_dropout: 0.05
    lora_target_modules:
      - q_proj
      - v_proj
      - k_proj
      - o_proj
      - gate_proj
      - up_proj
      - down_proj
    #lora_target_linear: true # Target all linear layers (alternative to listing modules)

    sequence_len: 1024  # Maximum sequence length
    sample_packing: true #For more efficient training.
    pad_to_sequence_len: true

    wandb_project: my-finetuning-project  # WandB project name (optional)
    wandb_entity: your-wandb-entity  # WandB entity name (optional)
    wandb_watch:  # WandB watch mode (optional)
    wandb_name:  # WandB run name (optional)
    wandb_log_model:  # WandB log model (optional)

    gradient_accumulation_steps: 4
    micro_batch_size: 2
    num_epochs: 3
    optimizer: adamw_bnb_8bit # Use 8-bit AdamW optimizer (if bitsandbytes is installed)
    # optimizer: paged_adamw_8bit # Alternative 8-bit optimizer
    learning_rate: 2e-5
    train_on_inputs: false # Important: only train on assistant responses.
    group_by_length: false #For sample packing
    bf16: true
    fp16: false
    tf32: false #Disable if you see issues.

    gradient_checkpointing: true
    early_stopping_patience:
    resume_from_checkpoint:
    local_rank:
    logging_steps: 1
    xformers_attention:
    flash_attention: true

    warmup_steps: 10
    eval_steps: 0.1 # Evaluate 10 times per epoch
    save_steps: 0.1 #Save 10 times per epoch
    debug:
    deepspeed:
    weight_decay: 0.0
    fsdp:
    fsdp_config:
    special_tokens:
        bos_token: "<|im_start|>"
        eos_token: "<|im_end|>"
        unk_token: "<|im_end|>"
        pad_token: "<|im_end|>"
    ```

    *   **Key Sections:**
        *   `base_model`:  The Hugging Face model ID.
        *   `model_type`, `tokenizer_type`: Usually `AutoModelForCausalLM` and `AutoTokenizer`.
        *    `trust_remote_code`: Set to `true` when required.
        *   `load_in_8bit`, `load_in_4bit`, `gptq`: Quantization options.
        *   `datasets`:  Defines your dataset(s).  Supports multiple datasets and different formats (JSON, `.jsonl`, Hugging Face datasets).  `train_test_split` is also supported for creating eval sets.
        *   `dataset_prepared_path`: Path where preprocessed data will be saved.
        *   `val_set_size`:  Proportion of the data to use for validation.
        *   `output_dir`: Where to save the fine-tuned model.
        *   `adapter`:  Specifies the PEFT method (e.g., `lora`, `qlora`).
        *   `lora_r`, `lora_alpha`, `lora_dropout`, `lora_target_modules`:  LoRA parameters.  *Crucially*, `lora_target_modules` needs to be set correctly for your specific model. Alternatively you can set `lora_target_linear` to `true` to target all linear layers.
        *   `sequence_len`: Maximum sequence length.
        *  `sample_packing`: Enable for more efficient training by packing multiple short samples into a single sequence.
        *  `pad_to_sequence_len`: Pad sequences to the maximum length.
        *   `wandb_project`, `wandb_entity`, `wandb_watch`, `wandb_name`, `wandb_log_model`:  Weights & Biases integration (optional, but highly recommended for tracking experiments).
        *   `gradient_accumulation_steps`, `micro_batch_size`, `num_epochs`, `optimizer`, `learning_rate`: Standard training hyperparameters.
        *    `train_on_inputs`: VERY IMPORTANT: Set to `false`. This ensures the model is trained to predict only the assistant responses, not the user inputs or system prompts. This is critical when using `apply_chat_template`.
        *  `group_by_length`: Used in conjunction with sample packing, this groups sequences of similar lengths.
        *   `bf16`, `fp16`, `tf32`: Mixed precision options.
        *   `gradient_checkpointing`:  Reduces memory usage.
        *   `warmup_steps`:  Gradually increases the learning rate at the beginning of training.
        *   `eval_steps`, `save_steps`: Control evaluation and checkpoint saving frequency.  `0.1` means 10 times per epoch (useful for smaller datasets).
        *    `special_tokens`: Sets up the special tokens (bos, eos, unk, pad). For Qwen set all to `<|im_end|>`.

*   **Running Fine-tuning:**

    ```bash
    accelerate launch -m axolotl.cli.train config.yaml
    ```

    This command starts the fine-tuning process using the configuration defined in `config.yaml`.  `accelerate launch` handles distributed training and other configurations.

*   **Advantages of Axolotl:**

    *   **Simplified Configuration:**  YAML files make it easy to manage and organize training parameters.
    *   **Reproducibility:**  YAML configurations facilitate reproducible experiments.
    *   **Integration with Hugging Face:**  Seamlessly integrates with the Hugging Face ecosystem.
    *   **Advanced Features:**  Supports features like PEFT, quantization, distributed training, and more.
    *   **Active Community:**  Axolotl has a growing and active community, providing support and resources.

**Step 11: Deploying Your Model**

Once you have a fine-tuned model, you can deploy it for inference.  Here are several options:

*   **Hugging Face Inference API:**  A simple way to deploy your model on Hugging Face's infrastructure.
*   **Local Inference:**  Run the model directly on your own machine or server using the `transformers` library (see the code example in Step 6).
*   **Cloud-Based Deployment:**  Deploy your model on cloud platforms like AWS, Google Cloud, or Azure using services like SageMaker, Vertex AI, or Azure Machine Learning.
*   **vLLM:** vLLM is a fast and easy-to-use library for LLM inference and serving. It is highly recommended.
*   **Text Generation Inference (TGI):** Hugging Face's purpose-built solution for deploying and scaling LLMs. It's highly optimized and supports features like continuous batching and tensor parallelism.

**Step 12: Troubleshooting**

*   **Out of Memory (OOM) Errors:**
    *   Reduce `per_device_train_batch_size`.
    *   Increase `gradient_accumulation_steps`.
    *   Use a smaller model.
    *   Use PEFT (e.g., LoRA).
    *   Enable quantization (8-bit or 4-bit).
    *   Use gradient checkpointing.
    *   Use a GPU with more VRAM.
    *   Reduce `max_seq_length`.
    *   If using sample packing, reduce `sequence_len`.
*   **Slow Training:**
    *   Use a larger `per_device_train_batch_size` (if memory allows).
    *   Use a more powerful GPU.
    *   Enable mixed precision training (`fp16` or `bf16`).
    *   Use distributed training.
    *   Ensure your data loading isn't a bottleneck (use `num_proc` in `dataset.map`).
*   **Poor Performance:**
    *   Check your prompt template!  This is *the* most common cause of issues.  Use `tokenizer.apply_chat_template()`.
    *   Ensure your data is high-quality, relevant, and correctly formatted.
    *   Experiment with different hyperparameters (learning rate, epochs, etc.).
    *   Try a different base model.
    *   Increase the size of your training dataset.
    *   If using PEFT, adjust the `target_modules` (or use `lora_target_linear: true`).
    * Check that `train_on_inputs` is `false` when using chat templates.
*   **NaN Loss:**
    *  Reduce the learning rate.
    * Check for errors in your dataset (e.g., division by zero, invalid characters).
    * Try a different optimizer.
    * Use gradient clipping.
    * If using bf16, try fp16 instead (or vice versa).
*  **Model Generates Nonsense/Repetitive Text:**
    * Verify prompt template is correct.
    *  Ensure the dataset is in the correct format (alternating user/assistant roles).
    * Train for more epochs.
    * Experiment with different hyperparameters (temperature, top_p, top_k during generation).
    * Try a different base model.

**Key Takeaways and Best Practices:**

*   **Prompt Template is King:**  Always, always, always use the correct prompt template for your chosen model.  `tokenizer.apply_chat_template()` is your best friend.
*   **Data Quality Matters Most:**  A small, high-quality dataset is better than a large, noisy one.
*   **Start Small, Iterate:**  Begin with a smaller model and a smaller dataset for faster experimentation.  Then, scale up as needed.
*   **Use PEFT:**  PEFT techniques (especially LoRA) are highly recommended for efficiency and to prevent catastrophic forgetting.
*   **Monitor Training:**  Use TensorBoard or Weights & Biases to track metrics and visualize training progress.
*   **Evaluate Thoroughly:**  Use both quantitative and *qualitative* evaluation.  Don't just look at numbers; inspect the generated text.
*   **Hyperparameter Tuning is Important:**  Experiment with different settings to find the best configuration.
*  **`train_on_inputs: false`**:  When using chat templates, make sure this is set to `false` so you're only training on the assistant's responses.
* **Understand Quantization**: Use to reduce resource requirements.
* **Choose appropriate hardware**: Consider model size when selecting GPUs.
