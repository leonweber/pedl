from collections import defaultdict
from operator import itemgetter
from typing import Optional, Dict, List, Set
from argparse import ArgumentParser
from pathlib import Path
import random

from openpyxl.styles import Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl import Workbook
import pandas as pd
import numpy as np
import requests


from pedl.utils import build_summary_table

np.random.seed(42)
random.seed(42)

ESEARCH_MAX_COUNT = 100000


def get_pmid_to_mesh_terms(mesh_terms: Set[str]) -> Dict[str, List[str]]:
    pmid_to_mesh_terms = defaultdict(list)
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    for mesh_term in mesh_terms:
        n_processed = 0
        result = requests.get(base_url, params={"db": "pubmed",
                                                "term": f"{mesh_term}[MESH]",
                                                "tool": "PEDL",
                                                "email": "weberple@hu-berlin.de",
                                                "retmode": "json",
                                                "retmax": ESEARCH_MAX_COUNT
                                                }
                              )
        total_count = int(result.json()["esearchresult"]["count"])
        n_processed += len(result.json()["esearchresult"]["idlist"])
        for pmid in result.json()["esearchresult"]["idlist"]:
            pmid_to_mesh_terms[pmid].append(mesh_term)

        while n_processed < total_count:
            result = requests.get(base_url, params={"db": "pubmed",
                                                    "term": f"{mesh_term}[MESH]",
                                                    "tool": "PEDL",
                                                    "email": "weberple@hu-berlin.de",
                                                    "retmode": "json",
                                                    "retmax": ESEARCH_MAX_COUNT,
                                                    "retstart": n_processed
                                                    }
                                  )
            n_processed += len(result.json()["esearchresult"]["idlist"])
            for pmid in result.json()["esearchresult"]["idlist"]:
                pmid_to_mesh_terms[pmid].append(mesh_term)

    return pmid_to_mesh_terms






def adjust_sheet_width(sheet):
    # Adjust column width
    for col in sheet.columns:
        max_length = 0
        column = col[0].column_letter  # Get the column name
        for cell in col:
            try:  # Necessary to avoid error on empty cells
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min((max_length + 2) * 1.2, 255)
        sheet.column_dimensions[column].width = adjusted_width


def _add_value_choices(sheet):
    # Incorrect
    sheet["O1"].value = "Wrong Genes"
    sheet["O2"].value = "Wrong PPA Type"
    sheet["O3"].value = "No PPA"

    data_validation = DataValidation(type="list", formula1="$O$1:$O$100")
    data_validation.add("J2:J10000")
    sheet.add_data_validation(data_validation)

    # Useless
    sheet["N1"].value = "Indirect Interaction"
    sheet["N2"].value = "Insufficient Evidence"
    sheet["N3"].value = "Wrong Organism"
    sheet["N4"].value = "Wrong Disease"
    sheet["N5"].value = "Mutation Required"
    sheet["N6"].value = "PTM Required"
    sheet["N7"].value = "Bound Proteins Required"

    data_validation = DataValidation(type="list", formula1="$N$1:$N$100")
    data_validation.add("K2:K10000")
    sheet.add_data_validation(data_validation)

def df_to_workbook(df: pd.DataFrame, ppa_dir: Path, topk:int  = 5,
                   threshold: float = 0.5, pmid_to_mesh_terms: Optional[Dict] = None,
                   annotation_mode: bool = False
                   ) -> Workbook:
    wb = Workbook()
    sheet = wb.active
    header = ["head", "association type", "tail", "text", "pubmed", "article score", "MESH terms"]
    if annotation_mode:
        header += ["correct", "useful", "why incorrect?", "why not useful?", "comment"]
    for cell, value in zip(sheet["A1":"L1"][0], header):
        cell.value = value
        cell.font = Font(bold=True)

    idx_next_free_row = 2
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            fname = ppa_dir / ((row["head"] + "_" + row["tail"]).upper() + ".txt")
        except TypeError:
            print("Warning: Skipping invalid row. Fix this before producing project results")
            continue
        assert fname.exists()
        pmid_to_score = defaultdict(float)
        pmid_to_fields = defaultdict(list)
        with fname.open() as f:
            for line in f:
                fields = line.strip().split("\t")
                if len(fields) > 1:
                    ppa_type, score, pmid, text, _ = fields
                    if float(score) < threshold or ppa_type != row["association type"]:
                        continue

                    if pmid_to_mesh_terms and pmid not in pmid_to_mesh_terms:
                        continue

                    pmid_to_score[pmid] += float(score)
                    pmid_to_fields[pmid].append(fields)

            for pmid, article_score in sorted(pmid_to_score.items(), key=itemgetter(1), reverse=True)[:topk]:
                fields = pmid_to_fields[pmid][0]
                ppa_type, score, pmid, text, _ = fields
                sheet[f"A{idx_next_free_row + 2}"].value = row["head"]
                sheet[f"C{idx_next_free_row + 2}"].value = row["tail"]
                sheet[f"B{idx_next_free_row + 2}"].value = row["association type"]
                sheet[f"D{idx_next_free_row + 2}"].value = text
                sheet[f"E{idx_next_free_row + 2}"].hyperlink = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                sheet[f"E{idx_next_free_row + 2}"].value = pmid
                sheet[f"B{idx_next_free_row + 2}"].value = row["association type"]
                sheet[f"E{idx_next_free_row + 2}"].style = "Hyperlink"
                sheet[f"F{idx_next_free_row + 2}"].value = article_score

                if pmid in pmid_to_mesh_terms:
                    sheet[f"G{idx_next_free_row + 2}"].value = ", ".join(pmid_to_mesh_terms[pmid])
                idx_next_free_row += 1

    if annotation_mode:
        _add_value_choices(sheet)
    adjust_sheet_width(sheet)

    return wb

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ppa_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=False)
    parser.add_argument("--mesh_terms", nargs="*")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--annotation", action="store_true")
    args = parser.parse_args()
    args.output: Path
    args.output = args.output.with_suffix("")

    pmid_to_mesh_terms = get_pmid_to_mesh_terms(set(args.mesh_terms))

    df_summary = build_summary_table(
        raw_dir=args.ppa_dir,
        score_cutoff=args.threshold
        )

    output_file = args.output.with_suffix(".xlsx")
    wb = df_to_workbook(df_summary, args.ppa_dir, threshold=args.threshold, topk=args.topk, pmid_to_mesh_terms=pmid_to_mesh_terms,
                        annotation_mode=args.annotation)
    wb.save(output_file)
