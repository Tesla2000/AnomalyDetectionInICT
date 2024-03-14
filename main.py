import subprocess

import scipy

from Config import Config


def main():
    for dataset_path in Config.datasets_path.iterdir():
        print(dataset_path.name)
        dataset = scipy.io.loadmat(dataset_path)
        print("Shape:", dataset['X'].shape)

    Config.tex_path.write_text(Config.base_tex_path.read_text().format("text").replace('`~', '{').replace('~`', '}'))
    subprocess.run(["pdflatex", Config.tex_path.absolute()])


if __name__ == '__main__':
    main()
