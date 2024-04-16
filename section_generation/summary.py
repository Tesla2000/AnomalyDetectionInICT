from pathlib import Path

from Config import Config


def summary():
    text = r"\section{Summary}" "\n"
    text += "which shows that a number of dimentions doesn't play more significant role in outliers detection than other parameter of dataset"
    Config.text_sections.joinpath(Path(__file__).with_suffix(".tex").name).write_text(
        text
    )
