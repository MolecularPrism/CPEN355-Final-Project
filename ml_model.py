from preprocess import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import pickle
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# -----------------------
# 5. TF-IDF
# -----------------------
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
baseline=True or False
'''
def train_model(texts, labels, train_iter=1000, baseline=False):

    if baseline:
        # ===== Logistic Regression (your original) =====
        X, vectorizer = vectorize_text(texts)

        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42
        )

        model = LogisticRegression(max_iter=train_iter)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        print(f"\nAccuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return model, vectorizer

    else:
        # ===== Transformer (DistilBERT) =====
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

        # 🔥 SPEEDUP: use subset
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

       
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model.to(device)
        print("Using device:", device)

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=1,                
            per_device_train_batch_size=8,     
            per_device_eval_batch_size=8,
            logging_steps=50
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = logits.argmax(axis=1)
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds)
            return {"accuracy": acc, "f1": f1}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        results = trainer.evaluate()

        print(f"\nAccuracy: {results['eval_accuracy']:.4f}")
        print(f"F1 Score: {results['eval_f1']:.4f}")

        return model, tokenizer

def save_model(model, vectorizer):
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    # save vectorizer
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


def predict(texts, model, vectorizer):
    # Transformer case
    if isinstance(vectorizer, DistilBertTokenizerFast):
        encodings = vectorizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        encodings = {k: v.to(device) for k, v in encodings.items()}
        model.to(device)

        with torch.no_grad():
            outputs = model(**encodings)

        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        return preds

    # Baseline case
    else:
        X = vectorizer.transform(texts)
        return model.predict(X)

def decode_prediction(pred):
    return "Positive" if pred == 1 else "Negative"