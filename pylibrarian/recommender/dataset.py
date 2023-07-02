from google.cloud import bigquery
import pandas as pd


class Dataset:
    def __init__(self, project: str = None):
        self.client = bigquery.Client()
        self.project = project
        self.threshold = 5 # Minimum number of mentions of a package.

    def load_data(self) -> pd.DataFrame:
        FILTER_QUERY = f"SELECT requirement, COUNT(*) FROM `{self.project}.pylibrarian.requirements` GROUP BY requirement HAVING COUNT(*) > 1"

        QUERY = (
            f"SELECT package, requirement FROM {self.project}.pylibrarian.requirements"
            f"WHERE requirement IN {FILTER_QUERY}"
            )
        query_job = self.client.query(QUERY, project=self.project)
        rows = query_job.result()
        return pd.DataFrame(
            [{"package": r.package, "requirement": r.requirement} for r in rows]
        )