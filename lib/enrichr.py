import json
import requests
import pandas as pd
import altair as alt

ENRICHR_BASE_URL = "http://amp.pharm.mssm.edu"
ENRICH_FIELD_NAMES = [
    "rank",
    "term_name",
    "pvalue",
    "z_score",
    "combined_score",
    "overlapping_genes",
    "adjusted_pvalue",
    "old_pvalue",
    "old_adjusted_pvalue",
]
DEFUALT_TOOLTIP_FIELDS = [
    "term_name",
    "pvalue",
    "z_score",
    "overlapping_genes",
    "adjusted_pvalue",
]


class EnrichrQuery:
    def __init__(self, gene_list, description, library="Enrichr"):
        self.gene_list = gene_list
        self.description = description
        self.library = library
        self.user_list_id = self._add_list(gene_list, description)
        self.enriched_output = {}

    def enrich(self, gene_set_databases):
        for db in gene_set_databases:
            self.enriched_output[db] = self._enrich_gene_list(db)

    def plot(
        self,
        database=None,
        pvalue_thresh=0.5,
        ranking_field="combined_score",
        color_field="adjusted_pvalue",
        tooltip_fields=DEFUALT_TOOLTIP_FIELDS,
        color_scheme="lightgreyteal",
        with_href=True,
    ):
        if len(self.enriched_output) == 0:
            raise Exception("No enrichement results. Please make query.")
        if database in self.enriched_output.keys():
            return self._make_barplot(
                database,
                self.enriched_output[database],
                pvalue_thresh,
                ranking_field,
                color_field,
                tooltip_fields,
                color_scheme,
                with_href,
            )
        barplots = [
            self._make_barplot(
                db,
                enr_res,
                pvalue_thresh,
                ranking_field,
                color_field,
                tooltip_fields,
                color_scheme,
                with_href,
            )
            for db, enr_res in self.enriched_output.items()
        ]
        return alt.vconcat(*barplots)

    def _make_barplot(
        self,
        db_name,
        enrich_result_df,
        pvalue_thresh,
        ranking_field,
        color_field,
        tooltip_fields,
        color_scheme,
        with_href,
    ):
        barchart = (
            alt.Chart(enrich_result_df.query(f"adjusted_pvalue < {pvalue_thresh}"))
            .mark_bar()
            .encode(
                x=f"{ranking_field}:Q",
                y=alt.Y(
                    "term_name:N",
                    sort=alt.SortField(field=ranking_field, order="descending"),
                    title=None,
                ),
                color=alt.Color(
                    f"{color_field}:Q",
                    scale=alt.Scale(scheme=color_scheme),
                    sort="descending",
                ),
                tooltip=tooltip_fields,
            )
            .properties(title=db_name)
        )
        if with_href:
            barchart = barchart.encode(href="url:N").transform_calculate(
                url="https://www.google.com/search?q=" + alt.datum.term_name
            )
        return barchart

    def _enrich_gene_list(self, database):
        url = (
            f"{ENRICHR_BASE_URL}/{self.library}/"
            f"enrich?userListId={self.user_list_id}"
            f"&backgroundType={database}"
        )
        response = requests.get(url)
        if not response.ok:
            raise Exception(
                f"Error fetching enrichment results from {database} database."
            )
        data = json.loads(response.text)
        enrich_result_df = pd.DataFrame(data[database], columns=ENRICH_FIELD_NAMES)
        return enrich_result_df

    def _add_list(self, gene_list, description):
        """Posts gene list with description to Enrichr and returns ID."""
        genes_str = "\n".join(gene_list)
        payload = {"list": (None, genes_str), "description": (None, description)}
        url = f"{ENRICHR_BASE_URL}/{self.library}/addList"
        response = requests.post(url, files=payload)
        if not response.ok:
            raise Exception("Error analyzing gene list")
        data = json.loads(response.text)
        return data["userListId"]

    def _view_list(self):
        """Returns gene list and description by ID from Enrichr library."""
        url = f"{ENRICHR_BASE_URL}/{self.library}/view?userListId={self.user_list_id}"
        response = requests.get(url)
        if not response.ok:
            raise Exception("Error getting gene list")
        data = json.loads(response.text)
        return data

    def __iter__(self):
        for name, enrich_result_df in self.enriched_output.items():
            yield name, enrich_result_df

    def __repr__(self):
        return (
            f"<EnrichrQuery: num_genes: {len(self.gene_list)}, "
            f"description: {self.description}, "
            f"library: {self.library}>"
        )
