from google.cloud import bigquery
import pandas as pd


class Dataset:
    def __init__(self, project: str = None):
        self.client = bigquery.Client()
        self.project = project

    def load_data(self) -> pd.DataFrame:
        QUERY = "SELECT package, requirement FROM the-sandbox-386618.pylibrarian.requirements"
        query_job = self.client.query(QUERY, project=self.project)
        rows = query_job.result()
        return pd.DataFrame(
            [{"package": r.package, "requirement": r.requirement} for r in rows]
        )
