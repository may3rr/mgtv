from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer, BitsAndBytesConfig, PeftModel, LoraConfig, get_linear_schedule_with_warmup
import torch, os

model_id = "facebook/nllb-200-3.3B"   #  [oai_citation:1‡Hugging Face](https://huggingface.co/facebook/nllb-200-3.3B?utm_source=chatgpt.com)
tokenizer = AutoTokenizer.from_pretrained(model_id, src_lang="zho_Hans", use_fast=False)
# LoRA 8bit
bnb = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, quantization_config=bnb, device_map="auto")

peft_cfg = LoraConfig(
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    r=8, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM")
model = PeftModel.from_pretrained(model, peft_cfg)

ds = load_dataset("json", data_files={"train":"data/finetune/*.json"}, split="train")
def tok(ex):
    ex = tokenizer(ex['中文'], text_target=ex['文本'], truncation=True, max_length=256)
    return ex
ds = ds.map(tok, remove_columns=['中文','文本'], batched=True)

args = TrainingArguments(
    output_dir="models/nllb_finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    num_train_epochs=1,
    lr_scheduler_type="constant",
    logging_steps=50,
    save_strategy="epoch"
)
trainer = Trainer(model=model, args=args,
                  train_dataset=ds,
                  data_collator=DataCollatorForSeq2Seq(tokenizer, model=model))
trainer.train()
model.save_pretrained("models/nllb_finetuned")
tokenizer.save_pretrained("models/nllb_finetuned")