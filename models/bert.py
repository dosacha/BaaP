# models/bert.py

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class BertClassifier:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.device = torch.device("cpu")  # << 이 줄 추가
        self.model.to(self.device)         # << 이 줄 추가

    def train(self, texts, labels):
        dataset = TextDataset(texts, labels, self.tokenizer)

        training_args = TrainingArguments(
            output_dir='./results/bert',
            per_device_train_batch_size=8,
            num_train_epochs=2,
            logging_steps=10,
            save_strategy="no",
            # evaluation_strategy="no"
            no_cuda=True  # 🔥 CPU 강제 사용!
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset
        )

        trainer.train()

    def predict_proba(self, texts):
        self.model.eval()
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # << 이 줄 추가
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs[:, 1].tolist()  # "AI"일 확률