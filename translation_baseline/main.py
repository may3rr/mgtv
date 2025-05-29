import os, subprocess, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--tag', default='testa')  # 可换成 dev/train
a = parser.parse_args()

asr_csv   = f"data/{a.tag}_asr.csv"
pred_csv  = f"data/{a.tag}_predict.csv"

subprocess.run(f"python scripts/run_asr.py  --csv_in data/{a.tag}.csv --csv_out {asr_csv}", shell=True, check=True)
subprocess.run(f"python scripts/run_translate.py --csv_in {asr_csv}   --csv_out {pred_csv}", shell=True, check=True)
print("🎉 全流程完成！提交文件:", pred_csv)