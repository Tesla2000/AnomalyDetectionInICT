from typing import Sequence


def sequence2latex(
    array: Sequence[Sequence],
    caption: str = None,
    label: str = None,
    placement: str = None,
    columns_setup: str = None,
):
    table_text = (
        r"\begin{table}" + ((r"[" + placement + r"]") if placement else "") + "\n"
        r"\centering"
        "\n"
        + (r"\caption{" + caption + "}" if caption is not None else "")
        + ((r"\label{" + label + "}") if label is not None else "")
        + "\n"
        + r"\begin"
        + r"{tabular}"
        + "{"
    )
    if len(set(map(len, array))) == 1:
        table_text += (
            (
                ("| r" + (len(array[0]) - 1) * " | l" + " |")
                if columns_setup is None
                else columns_setup
            )
            + "}\n"
            + r"\hline"
            + "\n"
            + (r" \\ \hline" + "\n").join(
                " & ".join(str(value) for value in row) for row in array
            )
        )
    else:
        table_text += (
            (
                ("| r" + (max(map(len, array)) - 1) * " l" + " |")
                if columns_setup is None
                else columns_setup
            )
            + "}\n"
            + r"\hline"
            + "\n"
            + (r" \\ \hline" + "\n").join(
                " & ".join(
                    (
                        ("\multicolumn{1}{c|}{" + str(value) + "}")
                        if not isinstance(value, list | tuple)
                        else (
                            "\multicolumn{"
                            + str(len(value))
                            + "}{c|}{"
                            + "".join(map(str, value))
                            + "}"
                        )
                    ).replace("}{", "}{|", 1)
                    for value in row
                )
                for row in array
            )
        )
    table_text += r" \\ \hline" + "\n" + r"\end{tabular}" "\n" + r"\end{table}" + "\n"
    return table_text
