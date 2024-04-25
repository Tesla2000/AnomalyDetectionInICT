from pathlib import Path

from Config import Config
from section_generation._set_seed import _set_seed


@_set_seed
def summary():
    text = r"\section{Summary}" "\n"
    text += "Analysis for final results contained in tables "r"\ref{table:running_times_and_results_dedicated} and \ref{table:running_times_and_results_supervised}"
    text += " can lead to conclusion that supervised methods tend to have advanced over unsupervised once in terms of accuracy. "
    text += "The behavior can be explained by access to additional information that unsupervised methods lack. "
    text += "In terms of performance elliptic envelope method turns out yo be much more computationally expensive. "
    text += "The time complexity of the method increases sharply with the number of dimensions taken into account with "
    text += "it being 5 to 10 time more time consuming than isolation forest on thyroid and pima dataset with 6 and 8 "
    text += "dimensions respectively and more than 500 times more demanding in musk with 166 dimensions. "
    text += "It can be concluded that in most highly-dimensional cases isolation forest should be run first "
    text += "and elliptic envelope only if the results are unsatisfactory. "
    text += "Other unsupervised methods namely Local Outlier Factor and KMeans both have an advantage of being "
    text += "extraordinary fast compared to their counterparts but turn out to be less robust, with KMean needing "
    text += "additional analysis with PCA to work properly, and failed to deliver "
    text += "results of higher accuracy that isolation forest and elliptic envelope on every turn."
    Config.text_sections.joinpath(Path(__file__).with_suffix(".tex").name).write_text(
        text
    )
