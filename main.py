import subprocess

from Config import Config
from section_generation.summary import summary
from section_generation.analysis import analysis


def main():
    # write_datasets()
    # anomaly_detection_methods()
    # supervised_methods()
    # analysis()
    # summary()
    subprocess.run(["pdflatex", Config.tex_path.absolute()])


if __name__ == "__main__":
    main()
