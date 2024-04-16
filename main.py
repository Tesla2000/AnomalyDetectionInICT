import subprocess

from Config import Config
from section_generation.anomaly_detection_methods import anomaly_detection_methods
from section_generation.cluster_methods import cluster_methods
from section_generation.summary import summary
from section_generation.analysis import analysis
from section_generation.supervised_methods import supervised_methods
from section_generation.write_datasets import write_datasets

_ = anomaly_detection_methods, summary, analysis, supervised_methods, write_datasets, cluster_methods


def main():
    # write_datasets()
    # anomaly_detection_methods()
    # supervised_methods()
    cluster_methods()
    # analysis()
    # summary()
    subprocess.run(["pdflatex", Config.tex_path.absolute()])


if __name__ == "__main__":
    main()
