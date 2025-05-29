#!/usr/bin/env python3
# ----------------------------------------------------------
# 双路批量 ASR：Paraformer-Large + Whisper-Large-v3（8-bit）
# 1) Paraformer 批量推理，取 avg CER
# 2) 超阈值语音批量送 Whisper 再识别
# 输出 CSV 将新增 “语音识别结果” 列
# ----------------------------------------------------------

import os, argparse, torch, pandas as pd
from tqdm import tqdm
from modelscope.pipelines import pipeline as ms_pipeline
from modelscope.utils.constant import Tasks
from transformers import (AutoProcessor, AutoModelForSpeechSeq2Seq,
                          BitsAndBytesConfig, pipeline as hf_pipeline)

# ========= CLI =========
parser = argparse.ArgumentParser()
parser.add_argument("--csv_in",  required=True, help="输入 CSV（含 音频路径 列）")
parser.add_argument("--csv_out", required=True, help="输出 CSV（含 语音识别结果 列）")
parser.add_argument("--cer_thresh", type=float, default=0.15, help="Paraformer CER 阈值")
parser.add_argument("--para_batch", type=int, default=16, help="Paraformer batch")
parser.add_argument("--whis_batch", type=int, default=4,  help="Whisper   batch")
args = parser.parse_args()

# ========= Paraformer =========
para = ms_pipeline(
    task=Tasks.auto_speech_recognition,
    model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    model_revision="v2.0.4"
)

# ========= Whisper-v3 (8-bit) =========
processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
if processor.tokenizer.pad_token_id is None:          # 补 pad_token
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3",
    quantization_config=bnb_cfg,
    device_map="auto",
    torch_dtype=torch.float16,
)
wh_asr = hf_pipeline(
    "automatic-speech-recognition",
    model=whisper,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=args.whis_batch
)

def whisper_batch(paths):
    """批量 whisper 返回文本列表"""
    outs = wh_asr(paths, generate_kwargs={"task":"transcribe","language":"zh"})
    return [o["text"] for o in outs]

# ========= 读取 CSV & 修正路径 =========
df = pd.read_csv(args.csv_in)
root_dir = os.path.dirname(args.csv_in)
def to_full(p):
    p = str(p).lstrip('/')
    return p if os.path.isabs(p) else os.path.join(root_dir, p)
paths = df["音频路径"].astype(str).apply(to_full).tolist()

# ========= 主批量循环 =========
texts = []
for i in tqdm(range(0, len(paths), args.para_batch), desc="ASR", ncols=100):
    batch = paths[i:i+args.para_batch]

    # ① Paraformer 批量
    res_list = para(batch)            # list ⟨dict|list⟩

    # ② 判断回退
    need_paths, keep_texts = [], []
    for p, res in zip(batch, res_list):
        if isinstance(res, dict):
            cer  = res.get("cer", 0.0)
            text = res.get("text","")
        else:                         # 多段
            cer  = sum(s.get("cer",0) for s in res) / max(len(res),1)
            text = "".join(s.get("text","") for s in res)

        if cer and cer > args.cer_thresh:
            need_paths.append(p)
            keep_texts.append(None)   # 占位
        else:
            keep_texts.append(text)

    # ③ Whisper 批量回退
    if need_paths:
        wh_texts = whisper_batch(need_paths)
        wh_iter  = iter(wh_texts)
        keep_texts = [next(wh_iter) if t is None else t for t in keep_texts]

    texts.extend(keep_texts)

# ========= 保存 =========
df["语音识别结果"] = texts
df.to_csv(args.csv_out, index=False)
print(f"✅ ASR 完成 → {args.csv_out}")