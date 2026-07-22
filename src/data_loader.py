import requests
import json
import pandas as pd
from json_repair import repair_json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pyarabic.araby as araby

def clean_arabic_text(text):
    text = str(text)
    text = araby.strip_tashkeel(text)
    text = araby.normalize_alef(text)
    return text

def load_and_prepare_data(url="https://raw.githubusercontent.com/NoorBayan/Burhan/main/corpus/similes_data.json"):
    response = requests.get(url)
    fixed_json_string = repair_json(response.text)
    data = json.loads(fixed_json_string)

    records = []
    for item in data:
        ayah = item.get('metadata', {}).get('ayah_text_uthmani', '')
        similes = item.get('rhetorical_analysis', {}).get('similes', [])
        
        if not similes: continue
        
        for simile in similes:
            functions = simile.get('functions', [])
            if functions:
                func = functions[0].get('pragmatic_function_tage')
                # 🌟 دمج الآية مع التشبيه يعطي سياقاً أقوى للنموذج
                simile_text = simile.get('simile_text', '')
                combined_text = f"{ayah} [SEP] {simile_text}" if simile_text else ayah
                
                if func and ayah:
                    records.append({'text': combined_text, 'label_text': func})
                    break 

    df = pd.DataFrame(records)
    df['clean_text'] = df['text'].apply(clean_arabic_text)

    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label_text'])

    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df['label'])
    
    return train_df, test_df, label_encoder
