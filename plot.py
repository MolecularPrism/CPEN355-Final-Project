import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import os
os.makedirs("plots", exist_ok=True)


# 1. Plot model comparison (Accuracy + F1)
def plot_model_results(models, accuracy, f1_scores, filename_prefix="model"):
    x = range(len(models))

    plt.figure()
    plt.bar(x, accuracy)
    plt.xticks(x, models)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.savefig(f"plots/{filename_prefix}_accuracy.png")

    plt.figure()
    plt.bar(x, f1_scores)
    plt.xticks(x, models)
    plt.ylabel("F1 Score")
    plt.title("Model F1 Score Comparison")
    plt.savefig(f"plots/{filename_prefix}_f1.png")

    plt.close()


#Plot LSTM loss curve
def plot_lstm_loss(loss_history, filename="lstm_loss"):
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM Training Loss Curve")
    plt.savefig(f"plots/{filename}.png")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, filename="confusion_matrix", title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.title(title)
    plt.savefig(f"plots/{filename}.png")
    plt.close()