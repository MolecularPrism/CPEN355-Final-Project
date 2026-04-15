from preprocess import *
from ml_model import *
from plot import plot_model_results, plot_lstm_loss, plot_confusion_matrix

def main():
    texts, labels = run_preprocess(generate_dataset=False)

    results = []

    # 1. Logistic Regression
    model1, vec1, acc1, f11, preds1, y_true1 = train_model(texts, labels, model_option=1)

    results.append(("LogReg", acc1, f11))

    plot_confusion_matrix(y_true1, preds1, filename="logreg_cm")


    # 2. DistilBERT

    model2, vec2, acc2, f12, preds2, y_true2 = train_model(texts, labels, model_option=2)
    results.append(("DistilBERT", acc2, f12))

    plot_confusion_matrix(y_true2, preds2, filename="bert_cm")



    # 3. LSTM
    model3, vec3, acc3, f13, loss_history, preds3, y_true3 = train_model(texts, labels, model_option=3)
    results.append(("LSTM", acc3, f13))

    plot_confusion_matrix(y_true3, preds3, filename="lstm_cm")
    plot_lstm_loss(loss_history, filename="lstm_loss")



    # Extract for plotting
    models = [r[0] for r in results]
    accuracy = [r[1] for r in results]
    f1_scores = [r[2] for r in results]

    print("\nFinal Results:")
    for name, acc, f1 in results:
        print(f"{name}: Accuracy={acc:.4f}, F1={f1:.4f}")


    # Plot compariso
    plot_model_results(models, accuracy, f1_scores, filename_prefix="comparison")


if __name__ == "__main__":
    main()