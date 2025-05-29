# 🎙️ MTGV 2025 Translation Baseline (ASR + Translation)

本项目是芒果 TV 多语种翻译比赛的基线系统，基于 Paraformer、Whisper 和 NLLB 构建，完成从中文语音到中→英/泰/马文本翻译任务。

---

## 🗂️ 项目结构

translation_baseline/
├── main.py                 # 主运行入口
├── nllb.py                 # NLLB 翻译模块
├── scripts/
│   └── run_asr.py          # ASR 阶段（Paraformer + Whisper融合）
├── local_nllb_600m/        # 本地加载的 NLLB 模型目录
├── data/
│   ├── train.csv           # 训练集（中文 → 目标语）
│   ├── dev.csv             # 验证集
│   ├── testa.csv           # 评测集
│   └── alg_2025_audios_final/ # 对应音频文件夹
└── README.md               # 当前文件

---

## ⚙️ 安装依赖

建议使用 Python 3.10+，推荐 Conda 环境：

```bash
conda create -n mgtv python=3.10 -y
conda activate mgtv

# 安装 PyTorch（参考官网适配你的 CUDA）
pip install torch torchaudio

# 安装主要依赖
pip install -r requirements.txt

如无 requirements.txt，可参考：

transformers>=4.36
datasets
modelscope
funasr==1.2.6
tqdm
pandas
sentencepiece
jieba


⸻

📁 数据准备

将比赛提供的压缩包 alg_2025_audios_final.zip 解压至 data/ 目录下，确保结构如下：

data/
├── train.csv
├── dev.csv
├── testa.csv
└── alg_2025_audios_final/
    ├── 电视剧1 EP01_00_00_38,580_00_00_42,420.wav
    └── ...


⸻

🚀 如何运行

1. 语音识别 + 翻译整体运行（推荐）

python main.py --tag testa

将会执行：
	•	✅ Paraformer + Whisper 融合识别 → testa_asr.csv
	•	✅ NLLB 翻译 → testa_trans.csv

2. 单独运行 ASR

python scripts/run_asr.py --csv_in data/dev.csv --csv_out data/dev_asr.csv


⸻

🧠 模型说明

✅ ASR 阶段
	•	Paraformer-Large（FunASR）
	•	模型来源：modelscope
	•	精度高，推理快
	•	Whisper-Large-v3（HuggingFace）
	•	在 CER 超阈值时 fallback，融合鲁棒性好

融合策略：
	•	默认采用 Paraformer
	•	当 cer > 阈值（默认 0.15）时，切换 Whisper 识别结果

✅ 翻译阶段
	•	NLLB-200（600M / 3.3B）
	•	支持中→泰/英/马 等语种
	•	使用 train.csv 中的中-外对进行微调（支持）

⸻

📈 输出格式说明
	•	xxx_asr.csv：增加“语音识别结果”列
	•	xxx_trans.csv：增加“最终翻译结果”列

⸻

🗝️ Git 版本管理建议

请确保你使用 .gitignore 排除了以下内容：

# 数据 & 模型文件
*.zip
*.pt
*.ckpt
*.bin
data/alg_2025_audios_final/
*.cache/
local_nllb_600m/


⸻

🙌 致谢
	•	FunASR
	•	Whisper
	•	NLLB-200

