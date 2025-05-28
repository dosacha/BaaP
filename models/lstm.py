# models/lstm.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, 128)
        self.lstm = nn.LSTM(128, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

class SimpleTextDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.LongTensor(self.sequences[idx]), self.labels[idx]

class LSTMClassifier:
    def __init__(self):
        self.vectorizer = CountVectorizer(max_features=1000, token_pattern=r"\b\w+\b")
        self.model = None

    def _prepare_data(self, texts, labels):
        X = self.vectorizer.fit_transform(texts).toarray()
        dataset = SimpleTextDataset(X, labels)
        return DataLoader(dataset, batch_size=8, shuffle=True)

    def train(self, texts, labels):
        dataloader = self._prepare_data(texts, labels)
        input_dim = len(self.vectorizer.get_feature_names_out())
        self.model = LSTMNet(input_dim)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train()
        for epoch in range(2):
            for x, y in dataloader:
                pred = self.model(x)
                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def predict_proba(self, texts):
        X = self.vectorizer.transform(texts).toarray()
        X = torch.LongTensor(X)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            probs = torch.nn.functional.softmax(logits, dim=1)
        return probs[:, 1].numpy().tolist()