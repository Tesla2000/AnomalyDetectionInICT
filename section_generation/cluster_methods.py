from itertools import chain
from pathlib import Path
from time import time

import numpy as np
import scipy
from fig2latex import fig2latex
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from Config import Config
from section_generation._set_seed import _set_seed
from sequence2array import sequence2latex


@_set_seed
def cluster_methods():
    method_running_times = []
    text = "\section{Neighbourhood anomaly detection}"
    text += "\nChose KMean as a clusterization method tp determine whether a sample is inliner or an outlier."
    text += " The assumption that there are 2 clusters, one representing majority class and the other (smaller one representing minority class), was made."
    dataset_path = Config.datasets_path.joinpath('pima.mat')
    dataset = scipy.io.loadmat(dataset_path)
    dataset_name = dataset_path.with_suffix("").name
    x, y = dataset["X"], dataset["y"]
    del dataset
    method_running_times.append([dataset_name])
    model_types = (KMeans,)
    model = KMeans(n_clusters=2)
    start = time()
    x = normalize(x)
    model.fit(x)
    model.predict(x)
    majority_class = int(x.sum() > .5 * len(x))
    center0, center1 = model.cluster_centers_
    distance_factors = tuple(np.sum((sample - (center0 if majority_class else center1)) ** 2) / np.sum((sample - (center0 if not majority_class else center1)) ** 2) for sample in x)
    execution_time = time() - start
    fpr, tpr, thresholds = roc_curve(y, distance_factors)
    roc_auc = auc(fpr, tpr)
    method_running_times[-1].append(round(roc_auc, 2))
    method_running_times[-1].append(round(execution_time, 3))
    plt.plot(fpr, tpr, lw=2, label=f"{model_types[0].__name__} (area = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve of {dataset_name}")
    plt.legend()
    image_path = Config.image_path.joinpath(Path(__file__).name + dataset_name + ".png")
    plt.savefig(image_path)
    plt.clf()
    text += "\nThe ROC curve with results is presented in Figure "r"\ref{figure:" + dataset_name + r"_cluster}"":\n"
    text += fig2latex(
        image_path.relative_to(Config.root),
        placement="h",
        caption=f"ROC curve of {dataset_name} dataset",
        label=f"figure:{dataset_name}_cluster",
        text_width=1,
    )
    table_name = "table:running_times_and_results_cluster"
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
