import subprocess

import numpy as np
import scipy
from fig2latex import fig2latex
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, RocCurveDisplay
from sklearn.neighbors import LocalOutlierFactor

from Config import Config
import array_to_latex


def main():
    text = r'\section{Zbiory danych}'
    text += "\nWybrano następujące zbiory danych"
    text += " ".join(dataset_path.name for dataset_path in Config.datasets_path.iterdir())
    table_name = "table:wymiary_zbiorow"
    text += r" z wymiarami podanymi w tabeli \ref{" + table_name + "}."
    text += "\n"
    text += array_to_latex.to_ltx(np.array((("Zbiór", "Liczba próbek", "Liczba wymiarów"), *tuple(
        (dataset_path.with_suffix('').name, *scipy.io.loadmat(dataset_path)['X'].shape) for dataset_path in
        Config.datasets_path.iterdir()))), print_out=False, caption="Informacje ilościowe o badanych zbiorach",
                                  label=table_name)
    for dataset_path in Config.datasets_path.iterdir():
        dataset = scipy.io.loadmat(dataset_path)
        dataset_name = dataset_path.with_suffix('').name
        text += '\n'r'\section{Zbiór danych ' + dataset_name + '}\n'
        x, y = dataset['X'], dataset['y']
        del dataset
        model = LocalOutlierFactor(n_neighbors=35)
        method_name = type(model).__name__
        table_reference = f"table:{method_name}"
        y_pred = model.fit_predict(x)
        y_pred[y_pred == 1] = 0  # inliers
        y_pred[y_pred == -1] = 1  # outliers
        text += array_to_latex.to_ltx(confusion_matrix(y, y_pred), frmt='{:.0f}', print_out=False, caption=f"Macierz pomyłek metody {method_name}", label=table_reference)
        y_pred = model.negative_outlier_factor_
        RocCurveDisplay.from_predictions(
            y,
            y_pred,
            name="LOF",
            pos_label=0
        )
        plt.plot([0, 1], [0, 1], "k--")
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("LOF, Receiver Operating Characteristic")
        plt.legend()
        image_path = Config.image_path.joinpath(dataset_name + '.png')
        plt.savefig(image_path)
        text += '\n'
        text += fig2latex(image_path.relative_to(Config.root), caption=f"Krzywa ROC dla {dataset_name} i metody {method_name}", label=f'figure:{method_name}')

    Config.tex_path.write_text(Config.base_tex_path.read_text().format(text).replace('`~', '{').replace('~`', '}'))
    subprocess.run(["pdflatex", Config.tex_path.absolute()])


if __name__ == '__main__':
    main()
