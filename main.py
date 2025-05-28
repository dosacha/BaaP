# main.py

import os
from data_loader import load_data
from utils.preprocess import preprocess_texts
from models.bert import BertClassifier
from models.lstm import LSTMClassifier
from meta.meta_classifier import MetaClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def train_internal_models(train_texts, train_labels):
    print("✅ Training Internal Models...")

    # 내부 모델 인스턴스화
    bert = BertClassifier()
    lstm = LSTMClassifier()

    # 모델 학습
    bert.train(train_texts, train_labels)
    lstm.train(train_texts, train_labels)

    # 예측 확률 (soft output)
    bert_probs = bert.predict_proba(train_texts)
    lstm_probs = lstm.predict_proba(train_texts)

    # 예측 클래스 (hard output)
    bert_classes = [int(p > 0.5) for p in bert_probs]
    lstm_classes = [int(p > 0.5) for p in lstm_probs]

    # 확률 + 클래스 결합한 feature 생성
    voting_features = list(zip(bert_probs, lstm_probs, bert_classes, lstm_classes))

    return voting_features, bert, lstm

def main():
    print("🚀 AI 텍스트 생성 판별기 실행 시작")

    # 1. 데이터 로딩 및 분할
    # texts, labels = load_data("data/train.csv")
    # train_texts, test_texts, train_labels, test_labels = train_test_split(
    #     texts, labels, test_size=0.2, stratify=labels, random_state=42
    # )
    train_texts, train_labels = load_data("data/extended_train.csv")
    train_texts, test_texts, train_labels, test_labels = train_test_split(
    train_texts, train_labels, test_size=0.2, stratify=train_labels, random_state=42
    )
    print(f"📊 훈련 샘플 수: {len(train_texts)}")

    # 2. 전처리
    train_texts = preprocess_texts(train_texts)
    test_texts = preprocess_texts(test_texts)

    # 3. 내부 모델 학습 및 feature 생성
    voting_features, bert_model, lstm_model = train_internal_models(train_texts, train_labels)

    # 4. 메타 분류기 학습
    meta_clf = MetaClassifier(method="logreg")  # method 변경 가능
    meta_clf.train(voting_features, train_labels)
    print("✅ 메타 분류기 학습 완료")

    # 5. 모델 저장
    joblib.dump(meta_clf, "models/meta_classifier.pkl")

    # 6. 테스트 데이터에 대해 메타 분류기 예측
    bert_probs = bert_model.predict_proba(test_texts)
    lstm_probs = lstm_model.predict_proba(test_texts)
    bert_classes = [int(p > 0.5) for p in bert_probs]
    lstm_classes = [int(p > 0.5) for p in lstm_probs]
    test_features = list(zip(bert_probs, lstm_probs, bert_classes, lstm_classes))

    predictions = meta_clf.predict(test_features)

    # 7. 평가 지표 출력
    print("🔍 최종 예측 결과:", predictions)
    print("📊 Accuracy:", accuracy_score(test_labels, predictions))
    print("📊 Precision:", precision_score(test_labels, predictions))
    print("📊 Recall:", recall_score(test_labels, predictions))
    print("📊 F1 Score:", f1_score(test_labels, predictions))

if __name__ == "__main__":
    main()