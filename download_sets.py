from download import download

from Config import Config


def download_sets():
    musk_path = Config.datasets_path.joinpath("musk.mat")
    download("https://www.dropbox.com/s/we6aqhb0m38i60t/musk.mat?dl=0", musk_path)
    pima_path = Config.datasets_path.joinpath("pima.mat")
    download("https://www.dropbox.com/s/mvlwu7p0nyk2a2r/pima.mat?dl=0", pima_path)
    thyroid_path = Config.datasets_path.joinpath("thyroid.mat")
    download("https://www.dropbox.com/s/bih0e15a0fukftb/thyroid.mat?dl=0", thyroid_path)