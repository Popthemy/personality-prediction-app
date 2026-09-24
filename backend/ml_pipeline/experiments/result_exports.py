"""Compact Excel export for the five binary metrics and shared thresholds."""

from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


METRICS = (
    ("Accuracy", "accuracy"),
    ("Precision", "precision"),
    ("Recall (Sensitivity)", "recall"),
    ("Specificity", "specificity"),
    ("F1-Score", "f1_score"),
)

EXPERIMENT_ORDER = (
    ("E1", "lasso_baseline"),
    ("E2", "lasso_baseline_gan"),
    ("E3", "lasso_qlearn"),
    ("E4", "lasso_qlearn_gan"),
    ("E5", "lstm_baseline"),
    ("E6", "lstm_baseline_gan"),
    ("E7", "lstm_qlearn"),
    ("E8", "lstm_qlearn_gan"),
)


def _cell(ref, value, style=0):
    if value is None:
        return f'<c r="{ref}" s="{style}"/>'
    if isinstance(value, (int, float)):
        return f'<c r="{ref}" s="{style}"><v>{value}</v></c>'
    return (f'<c r="{ref}" s="{style}" t="inlineStr"><is><t>'
            f'{escape(str(value))}</t></is></c>')


def export_metric_tables(path: Path, results, thresholds, trait_keys, trait_names):
    """Write all 25 binary-metric tables on one immediately visible worksheet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conditions = [key for _, key in EXPERIMENT_ORDER]
    if set(conditions) != set(results):
        raise ValueError("The comparison workbook requires all eight experiments.")
    rows = []

    def add(row_number, values, style=0, height=None):
        cells = ''.join(_cell(f'{chr(65 + column)}{row_number}', value, style)
                        for column, value in enumerate(values))
        size = f' ht="{height}" customHeight="1"' if height else ''
        rows.append(f'<row r="{row_number}"{size}>{cells}</row>')

    add(1, ["Binary classification results — all metrics"], 1)
    add(2, ["25 tables: five thresholds for each of five metrics"])
    add(3, ["Metric score scale: 0–1 (0.8 = 80%)"])
    add(4, ["The same validation BFI threshold applies to truth and prediction"])
    for metric_index, (metric_name, metric_key) in enumerate(METRICS):
        section_top = 6 + metric_index * 53
        add(section_top, [metric_name], 1)
        for index, threshold in enumerate(thresholds):
            top = section_top + 2 + index * 10
            add(top, [f"{metric_name} at threshold {threshold:.3f}"], 1)
            add(top + 1, ["Metric score scale: 0–1 (0.8 = 80%)"])
            add(top + 2, ["Trait", "Threshold", *[f"{label} ({key.replace('_', ' ')})" for label, key in EXPERIMENT_ORDER]], 1, 42)
            for ti, (trait_key, trait_name) in enumerate(zip(trait_keys, trait_names)):
                scores = []
                for condition in conditions:
                    per_trait = results[condition]["validation"]["per_trait"]
                    block = per_trait.get(trait_key, per_trait.get(trait_name))
                    if block is None:
                        raise ValueError(f"Missing {condition}/{trait_name} result")
                    match = next((entry for entry in block["threshold_sweep"]
                                  if abs(float(entry["threshold"]) - threshold) < 1e-8), None)
                    if match is None:
                        raise ValueError(f"Missing {condition}/{trait_key}/{threshold} result")
                    scores.append(match[metric_key])
                add(top + 3 + ti, [trait_name, threshold, *scores], 2)
    sheet_xml = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                 '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                 '<cols><col min="1" max="1" width="24" customWidth="1"/>'
                 '<col min="2" max="2" width="13" customWidth="1"/>'
                 '<col min="3" max="10" width="23" customWidth="1"/></cols>'
                 '<sheetData>' + ''.join(rows) + '</sheetData></worksheet>')
    types = ['<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
             '<Default Extension="xml" ContentType="application/xml"/>',
             '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
             '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>']
    types.append('<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
    workbook_sheets = '<sheet name="All metrics" sheetId="1" r:id="rId1"/>'
    rels = ('<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>')
    with ZipFile(path, "w", ZIP_DEFLATED) as book:
        book.writestr('[Content_Types].xml', '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">' + ''.join(types) + '</Types>')
        book.writestr('_rels/.rels', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        book.writestr('xl/workbook.xml', '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>' + workbook_sheets + '</sheets></workbook>')
        book.writestr('xl/_rels/workbook.xml.rels', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + rels + '</Relationships>')
        book.writestr('xl/styles.xml', '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><numFmts count="1"><numFmt numFmtId="164" formatCode="0.000"/></numFmts><fonts count="2"><font><sz val="11"/></font><font><b/><sz val="11"/></font></fonts><fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="3"><xf fontId="0" fillId="0" borderId="0" xfId="0"/><xf fontId="1" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment wrapText="1" vertical="center"/></xf><xf fontId="0" fillId="0" borderId="0" xfId="0" numFmtId="164" applyNumberFormat="1"/></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>')
        book.writestr('xl/worksheets/sheet1.xml', sheet_xml)
    return path
