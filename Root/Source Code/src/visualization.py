import matplotlib.pyplot as plt

def plot_tradeoff(results):
    acc = [r["accuracy"] for r in results]
    dpd = [r["dpd"] for r in results]

    plt.figure()
    plt.scatter(acc, dpd)
    plt.xlabel("Accuracy")
    plt.ylabel("DPD")
    plt.title("Fairness vs Accuracy")
    plt.savefig("fairness_vs_accuracy.png")
    plt.close()


def plot_bar(results, labels, key, filename):
    values = [r[key] for r in results]

    plt.figure()
    plt.bar(labels, values)
    plt.xticks(rotation=45)
    plt.title(key)
    plt.savefig(filename)
    plt.close()