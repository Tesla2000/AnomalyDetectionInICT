from typing import Sequence


def sequence2array(array: Sequence[Sequence], caption: str = None, label: str = None, placement: str = None):
    return r"\begin{table}" + (
        (r"[" + placement + r"]") if placement else "") + "\n"r"\centering""\n" + (r"\caption{" + caption + "}" if caption is not None else "") + (
        (r"\label{" + label + "}") if label is not None else "") + "\n" + r"\begin" + r"{tabular}" + "{" + (len(
        array[0]) * " | c" + " |") + "}\n" + r"\hline" + "\n" + (r" \\ \hline" + "\n").join(
        " & ".join(str(value) for value in row) for row in array) + r" \\ \hline" + "\n"r"\end{tabular}""\n"r"\end{table}""\n"
