from pathlib import Path

import numpy as np
import scipy
from fig2latex import fig2latex
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA

from Config import Config
from sklearn.preprocessing import normalize



def low_performance_kmeans():
    text = "\subsection{Low performace KMeans}\n"
    text += "The PCA was conducted on pima dataset to investigate the source of KMean low performance."
    dataset_path = next(Config.datasets_path.glob("pima*"))
    dataset = scipy.io.loadmat(dataset_path)
    x, y = dataset["X"], dataset["y"].flatten()
    del dataset
    pca_solver = PCA(2, random_state=Config.random_state)
    principal_components = pca_solver.fit_transform(x)
    x_coor = principal_components[:, 0]
    y_coor = principal_components[:, 1]
    plt.scatter(x_coor[np.where(y == 0)], y_coor[np.where(y == 0)], label="Inliners")
    plt.scatter(x_coor[np.where(y == 1)], y_coor[np.where(y == 1)], label="Outliers")
    image_name = "pima_pcs.png"
    image_path = Config.image_path.joinpath(image_name)
    plt.legend()
    plt.savefig(image_path)
    plt.clf()
    text += fig2latex(
        image_path.relative_to(Config.root),
        placement="h",
        caption="2 element principal component analysis of pima",
        label=f"figure:{image_name}",
    )
    text += r"The results are presented of figure \ref{figure:" + image_name + "}. "
    text += "As can be seem the separation is not as easy as in musk and the only noticeable distinction between"
    text += " outliers and inliners in grouping of the former in the center of second variables values and lower-end of "
    text += f"first variable values. With the number of dimensions at 8 and {round(sum(pca_solver.explained_variance_ratio_), 3)} of explained variance ratio"
    text += f" it is reasonable to assume that better results may be possible to obtain with distances to centroid as a metric."
    text += "\n\n To Similarly as in musk samples were scored for outlierness, this time with use of Mahalanobis. "
    model = EllipticEnvelope()
    model.fit(x)
    predictions = model.score_samples(x)
    predictions -= min(predictions)
    predictions /= max(predictions)
    cmap = plt.cm.get_cmap('RdYlGn')
    norm = Normalize(vmin=min(predictions), vmax=max(predictions))
    plt.scatter(x_coor, y_coor, c=predictions, cmap=cmap, norm=norm, alpha=0.8)
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='Inliner'),
                      plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Anomaly')]
    plt.legend(handles=legend_handles)
    image_name = "pima_score_mahalanobis.png"
    image_path = Config.image_path.joinpath(image_name)
    plt.savefig(image_path)
    plt.clf()
    text += fig2latex(
        image_path.relative_to(Config.root),
        placement="h",
        caption="EllipticEnvelope score of outlierness",
        label=f"figure:{image_name}",
    )
    text += r"The results presented at \ref{figure:" + image_name + "} explain why the results of EllipticEnvelope are superion to once obtained with KMean"
    Config.text_sections.joinpath(Path(__file__).with_suffix(".tex").name).write_text(
        text
    )
