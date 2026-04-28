import matplotlib.pyplot as plt
import os

def plot_tradeoff(results, dataset_name):
    acc = [r["accuracy"] for r in results]
    dpd = [r["dpd"] for r in results]

    plt.figure()
    plt.scatter(acc, dpd)
    plt.xlabel("Accuracy")
    plt.ylabel("DPD")
    plt.title(f"Fairness vs Accuracy - {dataset_name}")
    plt.savefig(f"outputs/plots/{dataset_name}_tradeoff.png")
    plt.close()


def plot_bar(results, labels, key, filepath):
    values = [r[key] for r in results]

    plt.figure()
    plt.bar(labels, values)
    plt.xticks(rotation=45)
    plt.title(key)
    plt.savefig(filepath)
    plt.close()