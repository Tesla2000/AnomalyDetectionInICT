from functools import wraps
from itertools import chain
from pathlib import Path
from time import time

import scipy
from fig2latex import fig2latex
from matplotlib import pyplot as plt
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor

from Config import Config
from section_generation._set_seed import _set_seed
from sequence2array import sequence2latex
from operator import attrgetter


@_set_seed
def anomaly_detection_methods():
    method_running_times = []
    text = "\section{Anomaly detection methods}\n"
    model_types = (
        wraps(LocalOutlierFactor)(lambda: LocalOutlierFactor(novelty=True)),
        EllipticEnvelope,
        IsolationForest,
    )
    text += f"Performed anomaly detection using novelty detection methods of {', '.join(map(attrgetter('__name__'), model_types))}. Plotted the results on the ROC curves showed on figures: "
    references = []
    for dataset_path in Config.datasets_path.iterdir():
        dataset = scipy.io.loadmat(dataset_path)
        dataset_name = dataset_path.with_suffix("").name
        x, y = dataset["X"], dataset["y"]
        del dataset
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=Config.random_state
        )
        del x, y
        method_running_times.append([dataset_name])
        for model_type in model_types:
            model = model_type()
            start = time()
            model.fit(x_train)
            y_pred = model.predict(x_test)
            y_pred[y_pred == 1] = 0
            y_pred[y_pred == -1] = 1
            method_name = model_type.__name__
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
        image_path = Config.image_path.joinpath(Path(__file__).name + dataset_name + ".png")
        plt.savefig(image_path)
        plt.clf()
        text += "\n"
        text += fig2latex(
            image_path.relative_to(Config.root),
            placement="h",
            caption=f"ROC curve of {dataset_name} dataset",
            label=f"figure:{dataset_name}_dedicated",
            text_width=1,
        )
        references.append(r"\ref{figure:" + dataset_name + "_dedicated}")
    text += ' ,'.join(references) + "."
    table_name = "table:running_times_and_results_dedicated"
    method_running_times = [
                               ["", *tuple((model_type.__name__, "") for model_type in model_types)],
                               [""]
                               + list(
                                   chain.from_iterable((("AUC", "Execution time [s]")) for _ in model_types)
                               ),
                           ] + method_running_times
    text += (
        "\n\nMethod running times and results are summed up in Table "
        r"\ref{" + table_name + "}:\n"
    )
    text += sequence2latex(
        method_running_times,
        label=table_name,
        placement="h",
        caption="AUC score and execution time of given method on dataset",
    )
    Config.text_sections.joinpath(Path(__file__).with_suffix(".tex").name).write_text(
        text
    )
