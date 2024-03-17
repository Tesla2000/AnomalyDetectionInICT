import subprocess
from itertools import chain
from time import time

import scipy
from fig2latex import fig2latex
from matplotlib import pyplot as plt
from scipy.stats import jarque_bera
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

from Config import Config
from sequence2array import sequence2latex


def main():
    text = r"\section{Zbiory danych}"
    text += "\nWybrano następujące zbiory danych"
    text += " ".join(
        dataset_path.name for dataset_path in Config.datasets_path.iterdir()
    )
    table_name = "table:wymiary_zbiorow"
    text += r" z wymiarami podanymi w tabeli \ref{" + table_name + "}."
    text += "\n"
    text += sequence2latex(
        (
            ("Zbiór", "Liczba próbek", "Liczba wymiarów", "Liczba outlierów"),
            *tuple(
                (
                    dataset_path.with_suffix("").name,
                    *(dataset := scipy.io.loadmat(dataset_path))["X"].shape,
                    int(dataset["y"].sum()),
                )
                for dataset_path in Config.datasets_path.iterdir()
            ),
        ),
        caption="Informacje ilościowe o badanych zbiorach",
        label=table_name,
        placement="h",
    )
    method_running_times = []
    for dataset_path in Config.datasets_path.iterdir():
        dataset = scipy.io.loadmat(dataset_path)
        dataset_name = dataset_path.with_suffix("").name
        text += "\n" + Config.dataset_descriptions.joinpath(dataset_name).read_text()
        text += "\n" r"\section{Zbiór danych " + dataset_name + "}\n"
        x, y = dataset["X"], dataset["y"]
        del dataset
        pvalue = jarque_bera(x).pvalue
        text += f"\nData was checked for normality with Jarque-Bera test showing {'normality' if pvalue > .05 else 'anormality'} of independent variables with p-value={pvalue}"
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=Config.random_state
        )
        del x, y
        method_running_times.append([dataset_name])
        model_types = (
            LocalOutlierFactor,
            EllipticEnvelope,
            IsolationForest,
        )
        for model_type in model_types:
            if "novelty" in model_type.__init__.__code__.co_varnames:
                model = model_type(novelty=True)
            else:
                model = model_type()
            start = time()
            model.fit(x_train)
            y_pred = model.predict(x_test)
            y_pred[y_pred == 1] = 0  # inliers
            y_pred[y_pred == -1] = 1  # outliers
            method_name = model_type.__name__
            # table_reference = f"table:{method_name}"
            # text += sequence2array(confusion_matrix(y_test, y_pred), caption=f"Macierz pomyłek metody {method_name}", label=table_reference, placement='h')
            y_scores = -model.score_samples(x_test)
            execution_time = time() - start
            fpr, tpr, thresholds = roc_curve(y_test, y_scores)
            roc_auc = auc(fpr, tpr)
            method_running_times[-1].append(round(roc_auc, 2))
            method_running_times[-1].append(round(execution_time, 3))
            plt.plot(fpr, tpr, lw=2, label=f"{method_name} (area = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC curve of {dataset_name}")
        plt.legend()
        image_path = Config.image_path.joinpath(dataset_name + ".png")
        plt.savefig(image_path)
        plt.clf()
        text += "\n"
        text += fig2latex(
            image_path.relative_to(Config.root),
            placement="h",
            caption=f"Krzywa ROC dla {dataset_name} i metody {method_name}",
            label=f"figure:{method_name}",
        )
    table_name = "table:running_times_and_results"
    text += (
        "\n\nMethod running times and results are summed up in Table "
        r"\ref{" + table_name + "}:\n"
    )
    method_running_times = [
        ["", *tuple((model_type.__name__, "") for model_type in model_types)],
        [""]
        + list(
            chain.from_iterable((("AUC", "Execution time [s]")) for _ in model_types)
        ),
    ] + method_running_times
    text += sequence2latex(
        method_running_times,
        label=table_name,
        caption="AUC score and execution time of given method on dataset",
    )
    Config.tex_path.write_text(
        Config.base_tex_path.read_text()
        .replace("{", "`~")
        .replace("}", "~`")
        .replace("`~~`", "{}")
        .format(text)
        .replace("`~", "{")
        .replace("~`", "}")
    )
    subprocess.run(["pdflatex", Config.tex_path.absolute()])


if __name__ == "__main__":
    main()
