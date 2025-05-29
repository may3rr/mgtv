import argparse, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--csv_in'); parser.add_argument('--csv_out')
args = parser.parse_args()

model_dir = "models/nllb_finetuned"
tok = AutoTokenizer.from_pretrained(model_dir, src_lang="zho_Hans")
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, device_map="auto", load_in_8bit=True)
trans_pipe = pipeline("text2text-generation", model=model, tokenizer=tok,
                      max_length=256, device=0, batch_size=4)

lang2tag = {'泰语':'tha_Thai', '英语':'eng_Latn', '马来语':'zsm_Latn'}

df = pd.read_csv(args.csv_in)
outputs = []
for text, lang in zip(df['语音识别结果'], df['语言']):
    tgt = lang2tag.get(lang, 'eng_Latn')
    out = trans_pipe(text, tgt_lang=tgt)[0]['generated_text']
    outputs.append(out)
df['answer'] = outputs
df.to_csv(args.csv_out, index=False)
print(f"Translate done → {args.csv_out}")