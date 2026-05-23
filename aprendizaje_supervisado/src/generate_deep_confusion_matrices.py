#aprendizaje_supervisado\src\generate_deep_confusion_matrices.py
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPORTS_DIR = Path("reports")


def extract_class_names(report_text):
    class_names = []
    for line in report_text.splitlines():
        match = re.match(r"\s*([A-Z])\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+", line)
        if match:
            class_names.append(match.group(1))
    return class_names


def extract_confusion_matrix(report_text, n_classes):
    matrix_text = report_text.split("Matriz de Confusión:", 1)[1]
    values = [int(value) for value in re.findall(r"\d+", matrix_text)]
    return np.array(values, dtype=int).reshape(n_classes, n_classes)


def plot_confusion_matrix(conf_matrix, class_names, title, save_path):
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(conf_matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase verdadera")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    threshold = conf_matrix.max() / 2
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            value = conf_matrix[i, j]
            if value > 0:
                ax.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                    fontsize=6,
                )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(REPORTS_DIR, exist_ok=True)

    best_reports = {
        "VGG16 - Experimento 2": (
            REPORTS_DIR / "evaluation_report_deep2_vgg16.txt",
            REPORTS_DIR / "confusion_matrix_best_vgg16.png",
        ),
        "ResNet50 - Experimento 6": (
            REPORTS_DIR / "evaluation_report_deep6_resnet50.txt",
            REPORTS_DIR / "confusion_matrix_best_resnet50.png",
        ),
    }

    for title_suffix, (report_path, output_path) in best_reports.items():
        if not report_path.exists():
            raise FileNotFoundError(f"No se encontró el reporte requerido: {report_path}")

        report_text = report_path.read_text(encoding="utf-8")
        class_names = extract_class_names(report_text)
        conf_matrix = extract_confusion_matrix(report_text, len(class_names))

        plot_confusion_matrix(
            conf_matrix,
            class_names,
            title=f"Matriz de Confusión - {title_suffix}",
            save_path=output_path,
        )
        print(f"[+] Matriz generada: {output_path}")
