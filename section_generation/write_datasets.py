import scipy
from scipy.stats import jarque_bera

from Config import Config
from sequence2array import sequence2latex


def write_datasets():
    dataset_text = r"\section{Datasets}"
    dataset_text += "\nThe folowing datasets were chosen"
    dataset_text += " ".join(
        dataset_path.name for dataset_path in Config.datasets_path.iterdir()
    )
    table_name = "table:wymiary_zbiorow"
    dataset_text += r" with parameters presented in \ref{" + table_name + "}."
    dataset_text += "\n"
    dataset_text += sequence2latex(
        (
            (
                "Dataset",
                "Number of samples",
                "Number of dimensions",
                "Number of outliers",
            ),
            *tuple(
                (
                    dataset_path.with_suffix("").name,
                    *(dataset := scipy.io.loadmat(dataset_path))["X"].shape,
                    int(dataset["y"].sum()),
                )
                for dataset_path in Config.datasets_path.iterdir()
            ),
        ),
        caption="Numerical information on datasets",
        label=table_name,
        placement="h",
    )
    for dataset_path in Config.datasets_path.iterdir():
        dataset = scipy.io.loadmat(dataset_path)
        dataset_name = dataset_path.with_suffix("").name
        dataset_text += (
            "\n" + Config.dataset_descriptions.joinpath(dataset_name).read_text()
        )
        dataset_text += "\n" r"\subsection{Dataset " + dataset_name + "}\n"
        x, y = dataset["X"], dataset["y"]
        del dataset
        pvalue = jarque_bera(x).pvalue
        dataset_text += f"\nData was checked for normality with Jarque-Bera test showing {'normality' if pvalue > .05 else 'anormality'} of independent variables with p-value={pvalue}"
    Config.text_sections.joinpath("datasets.tex").write_text(dataset_text)
