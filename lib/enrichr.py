import json
import requests
import sys
import gseapy as gp

ENRICHR_BASE_URL = "http://amp.pharm.mssm.edu/Enrichr"
ENRICHR_URL = f"{ENRICHR_BASE_URL}/addList"
ENRICHR_URL_A = f"{ENRICHR_BASE_URL}/view?userListId="
DEFAULT_GENE_SETS = ["KEGG_2016", "KEGG_2013"]


def run_enrichr(
    gene_list, label, outdir="enrichr", cutoff=0.1, gene_sets=DEFAULT_GENE_SETS
):
    return gp.enrichr(
        gene_list=gene_list,
        gene_sets=gene_sets,
        outdir=f"enrichr/{cell_id}",
        cutoff=cutoff,  # test dataset, use lower value from range(0,1)
    )


## TODO: remove dependency on gseapy 
def request_enrichr(gene_list, description, enr_library, outdir):
    payload = {"list": (None, gene_list), "description": (None, description)}

    res = requests.post(ENRICHR_URL, files=payload)

    if not res.ok:
        raise Exception("Enrichr not able to process request.")

    job = json.load(response.text)

    user_list_id = job["userListId"]
