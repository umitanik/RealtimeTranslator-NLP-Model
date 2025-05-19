import matplotlib.pyplot as plt
import os


def plot_training_history(history, metrics=None, validation=True, save_path=None, show=True):

    if metrics is None:
        metrics = ['loss']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history[metric], label=f'Training {metric}')

        if validation and f'val_{metric}' in history.history:
            plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')

        plt.title(f'Training and Validation {metric.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt_path = f"{os.path.splitext(save_path)[0]}_{metric}.png"
            plt.savefig(plt_path)
            print(f"✅ {metric} plot saved: {plt_path}")

        if show:
            plt.show()
        else:
            plt.close()
