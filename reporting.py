"""
Report generation module
"""

import pandas as pd
import numpy as np

# import pdfkit
import base64


def num_fmt(x, f):
    """
    Formats input as string.

    Arguments:
        x               Any numerical or string value.
        f               The specified format

    Format options:
        - int           Integer, as string, with comma separated thousands
        - int_k         Integer, as string, of thousands (x/1000)
        - int_mm        Integer, of Millions (x/1000000)
        - float_1       Float, one decimal point
        - float_1_mm    Float, one decimal, of Millions (x/1000000)
        - float_2       Float, two decimal points
        - pct_1         Percent (x*100), one decimal point

    Returns:
        The formatted string.
    """
    if f is not None and f not in [
        "int",
        "int_k",
        "int_mm",
        "float_1_mm",
        "float_2",
        "pct_1",
    ]:
        raise ValueError(f"Format specified '{f}' is not a valid format.")
    if isinstance(x, str):
        return x
    if np.isnan(x):
        return ""
    if f == "int":
        return f"{x:,.0f}"
    if f == "int_k":
        return f"{x/1000:,.0f}k"
    if f == "int_mm":
        return f"{x/1000000:,.0f}MM"
    if f == "float_1":
        return f"{x:,.1f}"
    if f == "float_1_mm":
        return f"{x/1000000:,.1f}MM"
    if f == "float_2":
        return f"{x:,.2f}"
    if f == "pct_1":
        return f"{100*x:,.1f}%"

    return str(x)


class ReportTable:
    """
    Defines a table which can be formatted for replacement in a report.

    Arguments:
        df          The DataFrame table to be formatted
        formats     A dict of {column_name: format}, usin
    """

    def __init__(
        self,
        df: pd.DataFrame,
        formats: dict = None,
        align: dict = None,
        totals: dict = None,
        col_widths: dict = None,
    ):
        self.df = df
        self.formats = formats if formats else {}
        self.align = align if align else {}
        self.totals = totals if totals else {}
        self.col_widths = col_widths if col_widths else {}

    @property
    def with_totals(self):
        """
        Adds a total row with defined functions for each column.
        """
        df = self.df.copy()
        if not self.totals:
            return df
        new_row = {}
        for col in df.columns:
            f = self.totals.get(col)
            if f is None:
                new_row[col] = [""]
            elif f == "sum":
                new_row[col] = [df[col].sum()]
            else:
                new_row[col] = [f]

        df_new = pd.DataFrame.from_dict(new_row)
        df = pd.concat([df, df_new])

        return df

    @property
    def formatted(self):
        """
        Converts all columns to strings in specified formats.

        Arguments:
            None

        Returns:
            A DataFrame with all values formatted as strings.
        """
        df = self.with_totals.copy()
        for k, v in self.formats.items():
            df[k] = [num_fmt(x, v) for x in df[k]]

        return df

    def to_html(self):
        """
        Exports the table as HTML. Column names are used as the header row.
        """
        df = self.formatted.copy()
        columns = df.columns.to_list()

        # Set alignment as specified or infer from
        aligns = []
        for col in columns:
            col_val = self.align.get(col)
            if col_val == "left":
                aligns.append("")
            elif col_val == "middle" or col_val == "center":
                aligns.append("text-center")
            elif col_val == "right":
                aligns.append("text-end")
            else:
                # Try to infer from the first value in the column
                val = df[col].to_list()[0]
                val = "".join([x for x in str(val) if not x in ",.-% "])
                if val.isnumeric():
                    aligns.append("text-end")
                else:
                    aligns.append("")

        tbl = ""
        tbl += '<table class="table table-sm">'

        # Col Groups
        col_widths = [self.col_widths.get(c) for c in columns]
        col_widths = [x if x else 1 for x in col_widths]
        col_widths = [100 * x / sum(col_widths) for x in col_widths]

        tbl += "<colgroup>"
        for col_width in col_widths:
            tbl += f'<col style="width: {col_width}%;">'
        tbl += "</colgroup>"

        # Header
        tbl += '<thead class="table-dark">'
        for col in columns:
            tbl += f'<th class="fw-normal fs-6 text-center">{col.replace("_", " ").upper()}</th>'
        tbl += "</thead>"

        # Body
        tbl += "<tbody>"
        total_rows = len(df.index)
        row_num = 0
        for row in df.itertuples(index=False):
            tbl += "<tr>"
            row_num += 1
            for cell, a in zip(row, aligns):
                classes = a
                if row_num == total_rows and self.totals:
                    classes += " fw-bold"
                tbl += f'<td class="{classes}">{cell}</td>'
            tbl += "</tr>"
        tbl += "</tbody>"

        tbl += "</table>"
        return tbl


class ReportFigure:
    def __init__(self, raw_data, img_ext="png"):
        self.raw = raw_data
        self.img_ext = img_ext

    @property
    def base64(self):
        """
        Returns the base64 encoded image string
        """
        return base64.b64encode(self.raw).decode()

    def to_html(self):
        """
        Encodes image as Base64 and embeds it in HTML
        """
        s = '<img src="data:image/png;base64,'
        s += self.base64
        s += '" />'

        return s


class Report:
    """
    Creates a report by filling a template with a set of replacements.
    """

    def __init__(self, template_file, output_file, replacements):
        self.template_file = template_file
        self.output_file = output_file
        self.replacements = replacements

    @property
    def raw(self):
        with open(self.template_file, "r", encoding="utf8") as f:
            text = f.read()
            for k, v in self.replacements.items():
                if not isinstance(v, str):
                    v = str(v)
                text = text.replace("{{" + k + "}}", v)
        return text

    def run(self):
        """
        Runs and exports the report
        """

        ext = self.output_file.split(".")[-1]

        if ext == "html":
            with open(self.output_file, "w", encoding="utf8") as f:
                f.write(self.raw)

        # elif ext == "pdf":
        #     pdfkit.from_string(self.raw, self.output_file)
        else:
            raise ValueError(f"Output file extension {ext} not recognized.")
