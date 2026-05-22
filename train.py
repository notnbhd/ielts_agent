"""
IELTS Writing Tutor — Finetune Qwen3-4B
====================================================
Dataset : nlpatunt/D_Ielts_Writing_Task_2_Dataset
Model   : Qwen/Qwen3-4B-Instruct-2507
Method  : SFT + LoRA (full FP16, không QLoRA)
Target  : V100 32GB / RTX 3090 24GB

Chạy:
    pip install transformers trl peft accelerate datasets
    python train.py
"""

# ── 0. Env vars (đặt trước mọi import) ─────────────────────────────────────
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
os.environ["PYTORCH_ALLOC_CONF"]     = "expandable_segments:True"

# ── 1. Imports ───────────────────────────────────────────────────────────────
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ── 2. Config ────────────────────────────────────────────────────────────────
MODEL_NAME  = "Qwen/Qwen3-4B-Instruct-2507"
OUTPUT_DIR  = "./ielts-qwen"
ADAPTER_DIR = "./ielts-lora-adapter"

TRAIN_SIZE  = 5000
VAL_SIZE    = 400

# ── 3. GPU info ──────────────────────────────────────────────────────────────
print(f"CUDA available : {torch.cuda.is_available()}")
print(f"GPU count      : {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name} — {props.total_memory / 1024**3:.1f} GB")

# ── 4. System prompt ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert IELTS Writing examiner and coach with 10+ years of experience \
training candidates from Band 5 to Band 8+. You have deep knowledge of the official IELTS Band \
Descriptors (0–9) published by the British Council and IDP.

When evaluating a Task 2 essay, you MUST follow this exact structure:

---
## Overall Band Score: [X.0 or X.5]

## Criterion Scores
| Criterion | Band | Justification (1 sentence) |
|-----------|------|---------------------------|
| Task Achievement (TA) | X.0 | ... |
| Coherence & Cohesion (CC) | X.0 | ... |
| Lexical Resource (LR) | X.0 | ... |
| Grammatical Range & Accuracy (GRA) | X.0 | ... |

> Overall = average of the 4 criterion scores, rounded to nearest 0.5

---
## Detailed Feedback

### Task Achievement
- **Position & argument:** [Is the thesis clear? Does it directly answer ALL parts of the prompt?]
- **Development:** [Are ideas fully extended with examples/reasoning, or are they underdeveloped?]
- **Band descriptor match:** [Quote the specific descriptor that best fits this essay]

### Coherence & Cohesion
- **Overall structure:** [Introduction → Body → Conclusion logic]
- **Paragraph cohesion:** [Does each paragraph have a clear central topic?]
- **Linking devices:** [Identify 2–3 specific connectives used; flag overuse or mechanical repetition]

### Lexical Resource
- **Strengths:** [Cite 2–3 specific words/phrases that demonstrate range]
- **Errors:** [List specific spelling/collocation/word-choice errors with corrections]
- **Sophistication:** [Does vocabulary feel natural or memorized/formulaic?]

### Grammatical Range & Accuracy
- **Range:** [What structures are used? Complex sentences, conditionals, passive voice, etc.]
- **Errors:** [Quote up to 5 specific error sentences, then provide the corrected version]
- **Proportion:** [Estimate % of error-free sentences]

---
## Examiner's Priority Improvements (Top 3 only)
1. [Highest-impact change — be specific, not generic]
2. [Second change]
3. [Third change]

## Rewrite Example
Rewrite ONE weak paragraph from the essay at a Band 7+ level, then briefly explain what changed.
---

Scoring rules:
- Band scores per criterion must be in increments of 0.5 (e.g. 5.0, 5.5, 6.0)
- Never award Band 9 unless the essay is genuinely indistinguishable from a native expert writer
- If the essay is under 250 words, cap TA at Band 5 regardless of quality
- If you see memorized/template phrases (e.g. "In this day and age", "It is a well-known fact"), \
flag them explicitly under LR
- Do not inflate scores to be encouraging; accurate feedback serves the candidate better\
"""

# ── 5. Load tokenizer ────────────────────────────────────────────────────────
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── 6. Load & format dataset ─────────────────────────────────────────────────
print("Loading dataset...")
raw_dataset = load_dataset("nlpatunt/D_Ielts_Writing_Task_2_Dataset")
print(raw_dataset)

def format_sample(row):
    user_message = (
        f"Please evaluate the following IELTS Writing Task 2 essay.\n\n"
        f"**Prompt:** {row['prompt']}\n\n"
        f"**Essay:**\n{row['essay']}"
    )
    assistant_message = f"**Overall Band Score: {row['band_score']}**\n\n{row['evaluation']}"

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": assistant_message},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

train_dataset = raw_dataset["train"].map(
    format_sample, remove_columns=raw_dataset["train"].column_names
)
val_dataset = raw_dataset["validation"].map(
    format_sample, remove_columns=raw_dataset["validation"].column_names
)
test_dataset = raw_dataset["test"].map(
    format_sample, remove_columns=raw_dataset["test"].column_names
)

train_dataset = train_dataset.select(range(min(TRAIN_SIZE, len(train_dataset))))
val_dataset   = val_dataset.select(range(min(VAL_SIZE,   len(val_dataset))))

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# ── 7. Load model (full FP16, không QLoRA) ───────────────────────────────────
print("\nLoading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map={"": torch.cuda.current_device()},
    trust_remote_code=True,
    attn_implementation="sdpa",
)
model.config.use_cache = False
print(f"Model dtype    : {model.dtype}")
print(f"Device map     : {model.hf_device_map}")

# ── 8. VRAM check ────────────────────────────────────────────────────────────
for i in range(torch.cuda.device_count()):
    allocated = torch.cuda.memory_allocated(i) / 1024**3
    reserved  = torch.cuda.memory_reserved(i)  / 1024**3
    total     = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f"GPU {i} | Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB "
          f"| Total: {total:.2f}GB | Free: {total - reserved:.2f}GB")

# ── 9. LoRA ──────────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=128,
    lora_alpha=256,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── 10. Training callback ─────────────────────────────────────────────────────
class TrainingMonitorCallback(TrainerCallback):
    """Print loss + VRAM usage mỗi logging_steps."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step      = state.global_step
        loss      = logs.get("loss", "N/A")
        eval_loss = logs.get("eval_loss", None)
        lr        = logs.get("learning_rate", "N/A")

        vram_info = ""
        for i in range(torch.cuda.device_count()):
            used  = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            vram_info += f" GPU{i}: {used:.1f}/{total:.1f}GB"

        msg = f"Step {step:4d} | loss: {loss} | lr: {lr}"
        if eval_loss:
            msg += f" | eval_loss: {eval_loss}"
        msg += f" |{vram_info}"
        print(msg)

# ── 11. Trainer ───────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[TrainingMonitorCallback()],
    args=SFTConfig(
        # --- Core ---
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,        # effective batch = 32
        warmup_ratio=0.05,
        learning_rate=2e-4,
        lr_scheduler_type="cosine_with_restarts",

        # --- Precision ---
        fp16=True,
        bf16=False,

        # --- Sequence ---
        max_seq_length=2048,
        dataset_text_field="text",
        packing=True,

        # --- Optimization ---
        optim="adamw_torch",
        weight_decay=0.01,
        gradient_checkpointing=False,         # 32GB đủ, không cần → nhanh hơn
        dataloader_num_workers=8,
        dataloader_pin_memory=True,

        # --- Logging & saving ---
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        output_dir=OUTPUT_DIR,

        seed=42,
        report_to="none",
    ),
)

# ── 12. Train ─────────────────────────────────────────────────────────────────
print("\nStarting training...")
trainer_stats = trainer.train()

print("\n=== Training Complete ===")
print(f"Time  : {trainer_stats.metrics['train_runtime']:.1f}s "
      f"({trainer_stats.metrics['train_runtime'] / 3600:.2f}h)")
print(f"Speed : {trainer_stats.metrics['train_steps_per_second']:.2f} steps/s")
print(f"Loss  : {trainer_stats.metrics['train_loss']:.4f}")

# ── 13. Save adapter ──────────────────────────────────────────────────────────
print(f"\nSaving adapter to {ADAPTER_DIR}...")
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)

size = sum(
    os.path.getsize(os.path.join(ADAPTER_DIR, f))
    for f in os.listdir(ADAPTER_DIR)
)
print(f"Adapter saved! Total size: {size / 1024**2:.1f} MB")

# ── 14. Quick inference test ──────────────────────────────────────────────────
TEST_PROMPT = (
    "Some people think that universities should provide graduates with the knowledge "
    "and skills needed in the workplace. Others think that the true function of a "
    "university should be to give access to knowledge for its own sake. "
    "Discuss both views and give your own opinion."
)
TEST_ESSAY = (
    "Nowadays, the purpose of university education is a topic of much debate. "
    "While some argue that universities should focus on practical workplace skills, "
    "others believe that the pursuit of knowledge for its own sake is more important. "
    "In my opinion, a balance between both approaches is ideal.\n\n"
    "On the one hand, universities that prioritize vocational training produce graduates "
    "who are immediately productive in the workforce. Employers benefit from hiring "
    "candidates who require minimal on-the-job training, which boosts economic efficiency. "
    "For example, medical and engineering programs have long combined theoretical knowledge "
    "with hands-on practice, resulting in highly competent professionals.\n\n"
    "On the other hand, knowledge pursued for its own sake fosters critical thinking, "
    "creativity, and intellectual independence. Subjects such as philosophy, literature, "
    "and pure mathematics may not have direct commercial applications, yet they cultivate "
    "the analytical skills that underpin innovation across all fields. Many breakthrough "
    "discoveries have emerged from curiosity-driven research with no initial practical goal.\n\n"
    "In conclusion, while workplace-relevant skills are undeniably important, universities "
    "should not abandon their role as centers of intellectual inquiry. The ideal graduate "
    "is one who is both professionally competent and capable of independent thought."
)

print("\n=== Quick Inference Test ===")
model.eval()

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user",   "content": (
        f"Please evaluate the following IELTS Writing Task 2 essay.\n\n"
        f"**Prompt:** {TEST_PROMPT}\n\n"
        f"**Essay:**\n{TEST_ESSAY}"
    )},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=2048,
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=1024,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True,
)
print(response)
