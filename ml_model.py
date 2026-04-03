from preprocess import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

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
def train_model(texts, labels, train_iter=1000, baseline=True):
    # STEP 3: Vectorize
    X, vectorizer = vectorize_text(texts)

    # STEP 4: Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )

    # STEP 5: Train model
    model = LogisticRegression(max_iter=train_iter)
    model.fit(X_train, y_train)

    # STEP 6: Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"\nAccuracy: {acc:.4f}") 

    return model, vectorizer

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
    X = vectorizer.transform(texts)  # IMPORTANT: transform, not fit_transform
    preds = model.predict(X)
    return preds

def decode_prediction(pred):
    return "Positive" if pred == 1 else "Negative"