import subprocess

import scipy
from fig2latex import fig2latex
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, RocCurveDisplay, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

from Config import Config
from sequence2array import sequence2array


def main():
    text = r'\section{Zbiory danych}'
    text += "\nWybrano następujące zbiory danych"
    text += " ".join(dataset_path.name for dataset_path in Config.datasets_path.iterdir())
    table_name = "table:wymiary_zbiorow"
    text += r" z wymiarami podanymi w tabeli \ref{" + table_name + "}."
    text += "\n"
    text += sequence2array((("Zbiór", "Liczba próbek", "Liczba wymiarów", "Liczba outlierów"), *tuple(
        (dataset_path.with_suffix('').name, *(dataset := scipy.io.loadmat(dataset_path))['X'].shape, int(dataset["y"].sum())) for dataset_path in
        Config.datasets_path.iterdir())), caption="Informacje ilościowe o badanych zbiorach",
                           label=table_name, placement='h')
    for dataset_path in Config.datasets_path.iterdir():
        dataset = scipy.io.loadmat(dataset_path)
        dataset_name = dataset_path.with_suffix('').name
        text += "\n" + Config.dataset_descriptions.joinpath(dataset_name).read_text()
        text += '\n'r'\section{Zbiór danych ' + dataset_name + '}\n'
        x, y = dataset['X'], dataset['y']
        del dataset
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=Config.random_state)
        del x, y
        for model_type in (LocalOutlierFactor, EllipticEnvelope, IsolationForest):
            if "novelty" in model_type.__init__.__code__.co_varnames:
                model = model_type(novelty=True)
            else:
                model = model_type()
            method_name = model_type.__name__
            table_reference = f"table:{method_name}"
            model.fit(x_train)
            y_pred = model.predict(x_test)
            y_pred[y_pred == 1] = 0  # inliers
            y_pred[y_pred == -1] = 1  # outliers
            text += sequence2array(confusion_matrix(y_test, y_pred), caption=f"Macierz pomyłek metody {method_name}", label=table_reference, placement='h')
            y_scores = model.score_samples(x_test)
            fpr, tpr, thresholds = roc_curve(y_test, y_scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve of {method_name} (area = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], "k--")
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("LOF, Receiver Operating Characteristic")
        plt.legend()
        image_path = Config.image_path.joinpath(dataset_name + '.png')
        plt.savefig(image_path)
        plt.clf()
        text += '\n'
        text += fig2latex(image_path.relative_to(Config.root), placement='h', caption=f"Krzywa ROC dla {dataset_name} i metody {method_name}", label=f'figure:{method_name}')

    Config.tex_path.write_text(Config.base_tex_path.read_text().replace('{', "`~").replace('}', "~`").replace('`~~`', '{}').format(text).replace('`~', '{').replace('~`', '}'))
    subprocess.run(["pdflatex", Config.tex_path.absolute()])


if __name__ == '__main__':
    main()
