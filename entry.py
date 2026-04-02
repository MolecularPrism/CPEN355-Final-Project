from preprocess import *
from ml_model import *


def main():
    # STEP 1: preprocess
    texts, labels = run_preprocess()

    # STEP 2: train
    model, vectorizer = train_model(texts, labels)

    # STEP 3: save
    save_model(model, vectorizer)

    # STEP 4 (optional): quick test
    sample = ["This product is amazing", "Worst purchase ever"]
    preds = predict(sample, model, vectorizer)

    print("\nSample predictions:")
    for text, pred in zip(sample, preds):
        print(f"{text} -> {pred}")

    


if __name__ == "__main__":
    main()