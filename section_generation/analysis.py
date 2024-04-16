from pathlib import Path

from Config import Config


def analysis():
    text = r"\section{Analysis}" "\n"
    text += (
        r"As showen in Table \ref{"
        + ""
        + "} the best results across all datasets were achieved by EllipticEnvelope and IsolationForest methods. "
        "The results varied a lot between datests with the worst obtained on pima dataset . "
        "Musk and thyroid scored similarly with, near-perfect results on effective methods and results being a bit worse on thyroid dataset."
    )
    Config.text_sections.joinpath(Path(__file__).with_suffix(".tex").name).write_text(
        text
    )
