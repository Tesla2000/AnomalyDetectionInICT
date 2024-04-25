from pathlib import Path

import numpy as np
import scipy
from fig2latex import fig2latex
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor

from Config import Config
from section_generation._set_seed import _set_seed


@_set_seed
def high_performance_musk():
    text = "\subsection{High performance on musk}\n"
    text += "The source of unusually high results on musk dataset was investigated through the use of PCA. "
    dataset_path = next(Config.datasets_path.glob("musk*"))
    dataset = scipy.io.loadmat(dataset_path)
    x, y = dataset["X"], dataset["y"].flatten()
    del dataset
    pca_solver = PCA(2, random_state=Config.random_state)
    principal_components = pca_solver.fit_transform(x)
    x_coor = principal_components[:, 0]
    y_coor = principal_components[:, 1]
    plt.scatter(x_coor[np.where(y == 0)], y_coor[np.where(y == 0)], label="Inliners")
    plt.scatter(x_coor[np.where(y == 1)], y_coor[np.where(y == 1)], label="Outliers")
    image_name = "musk_pcs.png"
    image_path = Config.image_path.joinpath(image_name)
    plt.legend()
    plt.savefig(image_path)
    plt.clf()
    text += fig2latex(
        image_path.relative_to(Config.root),
        placement="h!",
        caption="2 element principal component analysis of musk",
        label=f"figure:{image_name}",
        text_width=1,
    )
    text += r"The results are presented of Figure \ref{figure:" + image_name + "}:\n\n"
    text += "As can be seen PCA clearly separated claimed outliers from inliners. To further confirm that no specific feature is a matter of distinction corelation analysis was conducted."
    correlations = tuple(np.corrcoef(x[:, i], y)[0, 1] for i in range(x.shape[-1]))
    text += f"The maximal obtained correlation was {max(correlations)=:.2f} and the least {min(correlations)=:.2f} which showes that a combination of features caused high accuracy."
    lof = LocalOutlierFactor(novelty=True)
    lof.fit(x)
    predictions = lof.score_samples(x)
    predictions -= min(predictions)
    predictions /= max(predictions)
    cmap = plt.cm.get_cmap('RdYlGn')
    norm = Normalize(vmin=min(predictions), vmax=max(predictions))
    plt.scatter(x_coor, y_coor, c=predictions, cmap=cmap, norm=norm, alpha=0.8)
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='Inliner'),
                      plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Anomaly')]
    plt.legend(handles=legend_handles)
    image_name = "musk_score_lof.png"
    image_path = Config.image_path.joinpath(image_name)
    plt.savefig(image_path)
    plt.clf()
    text += fig2latex(
        image_path.relative_to(Config.root),
        placement="h",
        caption="LOF score of outlierness",
        label=f"figure:{image_name}",
        text_width=1,
    )
    text += r"Judging from the LOF scores of outlier detection presented of Figure \ref{figure:" + image_name + "}"
    text += "It can be deduced that LOF is not a valuable method of outlier detection of this task. Due to it marking"
    text += " samples at the edged of PCA clusters as outliers."
    Config.text_sections.joinpath(Path(__file__).with_suffix(".tex").name).write_text(
        text
    )
