# evaluate_dev.py
import pandas as pd, nltk, sys, numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
smooth = SmoothingFunction().method4

df = pd.read_csv('data/dev_predict.csv')
bleu2 = [sentence_bleu([ref.split()], hyp.split(), weights=(0.5,0.5), smoothing_function=smooth)
         for ref,hyp in zip(df['中文'], df['answer']) if isinstance(hyp,str)]
print("Avg BLEU2:", np.mean(bleu2))