from functools import wraps
from itertools import chain
from pathlib import Path
from time import time

import scipy
from fig2latex import fig2latex
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier

from Config import Config
from section_generation._set_seed import _set_seed
from sequence2array import sequence2latex


@_set_seed
def supervised_methods():
    method_running_times = []
    text = "\section{Supervised anomaly detection}\n"
    text += "Chose commonly used XGBoost and SVC ML models to determine if the sample in an inliner or an outlier.\n"
    dataset_path = Config.datasets_path.joinpath('pima.mat')
    dataset = scipy.io.loadmat(dataset_path)
    dataset_name = dataset_path.with_suffix("").name
    x, y = dataset["X"], dataset["y"]
    del dataset
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=Config.random_state
    )
    del x, y
    method_running_times.append([dataset_name])
    model_types = (
        wraps(SVC)(lambda: SVC(probability=True)),
        wraps(XGBClassifier)(lambda: XGBClassifier()),
    )
    for model_type in model_types:
        model = model_type()
        start = time()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_pred[y_pred == 1] = 0  # inliers
        y_pred[y_pred == -1] = 1  # outliers
        method_name = model_type.__name__
        y_scores = model.predict_proba(x_test)
        execution_time = time() - start
        fpr, tpr, thresholds = roc_curve(y_test, y_scores[:, 1])
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
    text += "\nThe ROC curve with results is presented in Figure "r"\ref{figure:" + dataset_name + r"_supervised}"":\n"
    text += fig2latex(
        image_path.relative_to(Config.root),
        placement="h",
        caption=f"ROC curve of {dataset_name} dataset",
        label=f"figure:{dataset_name}_supervised",
        text_width=1,
    )
    table_name = "table:running_times_and_results_supervised"
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
        placement="h",
        caption="AUC score and execution time of given method on dataset",
    )
    Config.text_sections.joinpath(Path(__file__).with_suffix(".tex").name).write_text(
        text
    )
