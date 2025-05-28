# data_loader.py

import pandas as pd

def load_data(file_path):
    # CSV 파일에서 텍스트와 라벨을 로드합니다.
    df = pd.read_csv(file_path)

    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV 파일에는 'text'와 'label' 열이 필요합니다.")

    texts = df['text'].astype(str).tolist()
    labels = df['label'].astype(int).tolist()

    return texts, labels