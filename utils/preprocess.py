# utils/preprocess.py

import re

def preprocess_texts(texts):
    """
    간단한 텍스트 전처리:
    - 소문자 변환
    - 특수문자 제거
    - 공백 정리
    """
    processed = []
    for text in texts:
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9가-힣\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        processed.append(text)
    return processed