from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,pipeline

#zsm_Latn 马来语   tha_Thai 泰语  eng_Latn 英语   zho_Hans  中文

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=False, src_lang="zho_Hans")
tokenizer.save_pretrained("./local_nllb_600m")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=False)
model.save_pretrained("./local_nllb_600m")

trans_tokenizer = AutoTokenizer.from_pretrained("./local_nllb_600m", use_auth_token=False, src_lang="zho_Hans")
trans_model = AutoModelForSeq2SeqLM.from_pretrained("./local_nllb_600m", use_auth_token=False)

ml_translator = pipeline('translation',model=trans_model,tokenizer=trans_tokenizer,src_lang="zho_Hans",tgt_lang="zsm_Latn",max_length=512,device='cuda:0')
th_translator =  pipeline('translation',model=trans_model,tokenizer=trans_tokenizer,src_lang="zho_Hans",tgt_lang="tha_Thai",max_length=512,device='cuda:0')
en_translator =  pipeline('translation',model=trans_model,tokenizer=trans_tokenizer,src_lang="zho_Hans",tgt_lang="eng_Latn",max_length=512,device='cuda:0')
