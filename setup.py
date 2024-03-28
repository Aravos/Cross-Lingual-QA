import sys
from transformers import MarianTokenizer, MarianMTModel, MarianConfig

model = {}
tokenizer = {}

def download_model(src, trg):
    model[(src, trg)] = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{src}-{trg}')
    model[(src, trg)].save_pretrained(f'./Models/{src}_{trg}')
    
    tokenizer[(src, trg)] = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{src}-{trg}')
    tokenizer[(src, trg)].save_pretrained(f'./Tokenizers/{src}_{trg}')


target_languages = ["en", "ar", "vi", "de", "zh"]

for src in target_languages:
    for trg in target_languages:
        if (src == "ar" and trg == "vi") or \
           (src == "ar" and trg == "zh") or \
           (src == "vi" and trg == "ar") or \
           (src == "vi" and trg == "zh") or \
           (src == "zh" and trg == "ar") or \
           (src == "zh" and trg == "fr"):
            continue
        if trg != src:
            download_model(src, trg)

print("Models downloaded")

model_names = {}
for src in target_languages:
    for trg in target_languages:
        if (src == "ar" and trg == "vi") or \
           (src == "ar" and trg == "zh") or \
           (src == "vi" and trg == "ar") or \
           (src == "vi" and trg == "zh") or \
           (src == "zh" and trg == "ar") or \
           (src == "zh" and trg == "fr"):
            continue
        if trg != src:
            model_name = f'Helsinki-NLP/opus-mt-{src}-{trg}'
            model_names[(src, trg)] = model_name

for (src, trg), model_name in model_names.items():
    config = MarianConfig.from_pretrained(model_names[(src, trg)])
    model[(src, trg)] = MarianMTModel.from_pretrained(model_names[(src, trg)], config=config)
    tokenizer[(src, trg)] = MarianTokenizer.from_pretrained(model_names[(src, trg)])
    
print("Models saved")
