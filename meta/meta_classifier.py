import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

class MetaClassifier:
    def __init__(self, method='lightgbm'):
        if method == 'lightgbm':
            self.model = lgb.LGBMClassifier()
        elif method == 'logreg':
            self.model = LogisticRegression()
        elif method == 'rf':
            self.model = RandomForestClassifier()
        else:
            raise ValueError(f"지원하지 않는 메타 분류기: {method}")

    def train(self, features, labels):
        X = np.array(features)  # ✅ numpy array 로 변환
        y = np.array(labels)
        self.model.fit(X, y)

    def predict(self, features):
        X = np.array(features)  # ✅ 예측 입력도 numpy array
        return self.model.predict(X)

    def predict_proba(self, features):
        X = np.array(features)
        return self.model.predict_proba(X)[:, 1]