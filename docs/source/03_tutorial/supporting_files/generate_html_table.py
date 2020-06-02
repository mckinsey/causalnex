import numpy as np
import pandas as pd


def to_html(df):
    # header = """<!DOCTYPE html>
    # <html>
    # <head>
    # <style>
    # table, th, td {
    #   border: 1px solid black;
    # }
    # </style>
    # </head>
    # <body>
    # """
    s = ["<table>"]

    # add header
    for el in df.columns:
        if el is np.nan:
            el = "--"
        else:
            s.append(f"<th>{el}</th>")

    for row in df.values:
        s.append("<tr>")

        first = True  # we apply "`" to the first element
        for el in row:
            if el is np.nan:
                el = ""

            if first:
                first = False
                el = ", ".join([f"`{e}`" for e in el.split(", ")])
            s.append(f"<td>{el}</td>")
        s.append("</tr>")
    return s


if __name__ == "__main__":
    graph = pd.read_excel("graphviz_att.xlsx", sheet_name="graph")
    edge = pd.read_excel("graphviz_att.xlsx", sheet_name="edge")
    node = pd.read_excel("graphviz_att.xlsx", sheet_name="node")

    with open("graph_att.html", "w") as f:
        f.writelines("\n".join(to_html(graph)))
    with open("edge_att.html", "w") as f:
        f.writelines("\n".join(to_html(edge)))
    with open("node_att.html", "w") as f:
        f.writelines("\n".join(to_html(node)))
