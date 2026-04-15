from preprocess import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import pickle
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)
    
class LSTMClassifier(nn.Module):
            def __init__(self, vocab_size):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
                self.lstm = nn.LSTM(128, 128, num_layers=2, bidirectional=True, batch_first=True)
                self.fc = nn.Linear(256, 2)

            def forward(self, x):
                x = self.embedding(x)
                _, (h, _) = self.lstm(x)
                h = torch.cat((h[-2], h[-1]), dim=1)
                return self.fc(h)

# TF-IDF
def vectorize_text(texts):
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X = vectorizer.fit_transform(texts)
    return X, vectorizer


'''
Vision is to have both baseline model and Neural Network Model
in this function and we choose what model to use based on 
model option
'''
def train_model(texts, labels, train_iter=1000, model_option=3):


    # OPTION 1 — BASELINE (LogReg)
    if model_option == 1:
        print("\n[Model] Logistic Regression")

        X, vectorizer = vectorize_text(texts)

        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42
        )

        model = LogisticRegression(max_iter=train_iter)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return model, vectorizer, acc, f1, preds, y_test


    # OPTION 2 — TRANSFORMER
    elif model_option == 2:
        print("\n[Model] DistilBERT")

        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

        # Fast subset
        train_texts = train_texts[:5000]
        train_labels = train_labels[:5000]
        test_texts = test_texts[:1000]
        test_labels = test_labels[:1000]

        train_encodings = tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=128
        )

        test_encodings = tokenizer(
            test_texts,
            truncation=True,
            padding=True,
            max_length=128
        )

        train_dataset = TextDataset(train_encodings, train_labels)
        test_dataset = TextDataset(test_encodings, test_labels)

        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )

        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        model.to(device)
        print("Using device:", device)

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=1,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            logging_steps=50,
            save_strategy="no",
            report_to="none",
            disable_tqdm=False
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )

        trainer.train()

        # Only do ONE pass on test set
        preds_output = trainer.predict(test_dataset)
        preds = preds_output.predictions.argmax(axis=1)

        acc = accuracy_score(test_labels, preds)
        f1 = f1_score(test_labels, preds)

        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return model, tokenizer, acc, f1, preds, test_labels


    # OPTION 3 — LSTM 
    elif model_option == 3:
        print("\n[Model] LSTM")

        # vocab 
        counter = Counter()
        for t in texts:
            counter.update(t.split())

        vocab = {"<pad>": 0, "<unk>": 1}
        for word, count in counter.most_common(20000):
            if count >= 3:
                vocab[word] = len(vocab)

        MAX_LEN = 200

        def encode(text):
            tokens = text.split()[:MAX_LEN]
            ids = [vocab.get(t, 1) for t in tokens]
            ids += [0] * (MAX_LEN - len(ids))
            return ids

        X = np.array([encode(t) for t in texts])
        y = np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.long),
            torch.tensor(y_train, dtype=torch.long)
        )

        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_test, dtype=torch.long),
            torch.tensor(y_test, dtype=torch.long)
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader  = DataLoader(test_dataset, batch_size=32)


        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        model = LSTMClassifier(len(vocab)).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # training 
        loss_history = []
        for epoch in range(10):
            model.train()
            total_loss = 0

            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)

                optimizer.zero_grad()
                outputs = model(xb)
                loss = criterion(outputs, yb)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            loss_history.append(avg_loss)

            print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

        # ---- eval ----
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)

                outputs = model(xb)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1  = f1_score(all_labels, all_preds)

        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return model, vocab, acc, f1, loss_history, all_preds, all_labels

    else:
        raise ValueError("model_option must be 1 (baseline), 2 (transformer), or 3 (lstm)")

def save_model(model, vectorizer):
    # Save vectorizer 
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # LSTM - save weights only
    if isinstance(vectorizer, dict):
        torch.save({
            "model_state": model.state_dict(),
            "vocab_size": len(vectorizer)
        }, "model.pt")

    # Transformer - save using HuggingFace
    elif isinstance(vectorizer, DistilBertTokenizerFast):
        model.save_pretrained("bert_model")
        vectorizer.save_pretrained("bert_model")

    # Baseline - pickle
    else:
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)


def load_model():

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    if isinstance(vectorizer, dict):
        checkpoint = torch.load("model.pt", map_location="cpu")
        model = LSTMClassifier(checkpoint["vocab_size"])
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

    elif isinstance(vectorizer, DistilBertTokenizerFast):
        model = DistilBertForSequenceClassification.from_pretrained("bert_model")
        vectorizer = DistilBertTokenizerFast.from_pretrained("bert_model")

    else:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)

    return model, vectorizer


def predict(texts, model, vectorizer):

    # Transformer
    if isinstance(vectorizer, DistilBertTokenizerFast):
        encodings = vectorizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )

        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}
        model.to(device)

        with torch.no_grad():
            outputs = model(**encodings)

        return torch.argmax(outputs.logits, dim=1).cpu().numpy()

    # LSTM
    elif isinstance(vectorizer, dict):
        MAX_LEN = 200

        def encode(text):
            tokens = text.split()[:MAX_LEN]
            ids = [vectorizer.get(t, 1) for t in tokens]
            ids += [0] * (MAX_LEN - len(ids))
            return ids

        X = torch.tensor([encode(t) for t in texts], dtype=torch.long)

        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        model.to(device)
        X = X.to(device)

        with torch.no_grad():
            outputs = model(X)

        return torch.argmax(outputs, dim=1).cpu().numpy()

    # Baseline
    else:
        X = vectorizer.transform(texts)
        return model.predict(X)

def decode_prediction(pred):
    return "Positive" if pred == 1 else "Negative"