import requests
import json
import xmlrpc
from time import sleep
from typing import List
from tqdm import tqdm
from pathlib import Path
from google.cloud import bigquery


class PypiScrapper:
    def __init__(
        self,
        output_file: str = None,
        package_source: str = "packages.txt",
        project: str = None,
    ):

        # Package source:
        self.package_source = (
            Path(__file__).parent.parent.parent / "data" / package_source
        )

        # Local output:
        self.output_file = output_file
        if output_file is not None:
            raise NotImplementedError("Local output file is not implemented.")

        # Database:
        self.client = bigquery.Client()
        self.project = project

        # Hyperparameters:
        self.sleep = 0.5
        self.packages = self.get_packages()

    def get_packages(self) -> List[str]:

        with open(self.package_source, "r") as f:
            packages = f.read().splitlines()

        done_packages = self.package_done_table()

        packages = [p for p in packages if p not in done_packages]

        print(f"Found {packages} packages.")

        return packages

    def package_done_local(self) -> List[str]:

        with open(self.output_file, "r") as f:
            done_packages = [l.split(";")[0] for l in f.readlines()]

        return done_packages

    def package_done_table(self) -> List[str]:
        QUERY = f"SELECT DISTINCT package FROM {self.project}.pylibrarian.requirements"
        query_job = self.client.query(QUERY, project=self.project)
        rows = query_job.result()
        return [r.package for r in rows]

    def get_requirements(self, package_name: str) -> List[str]:
        request = requests.get(f"https://pypi.org/pypi/{package_name}/json")
        if request.status_code != 200:
            print(f"Request failed for {package_name}", request.status_code)
            return []
        else:
            content = request.content
            requires_dist = json.loads(content)["info"]["requires_dist"]
            if requires_dist is None:
                requires_dist = []
            return [r.split(" ")[0] for r in requires_dist]

    def scrap(self):
        pbar = tqdm(self.packages)
        for package in pbar:
            pbar.set_description(f"Processing {package}")
            requirements = self.get_requirements(package)
            self.write_row(package, requirements)
            sleep(self.sleep)

    def write_row(self, package: str, requirements: List[str]):

        rows = [
            {"package": package, "requirement": requirement}
            for requirement in requirements
        ]

        if len(rows) > 0:
            errors = self.client.insert_rows_json(
                f"{self.project}.pylibrarian.requirements", rows
            )

        if self.output_file is not None:
            row = f"{package};{requirements}"
            with open(self.output_file, "a") as f:
                f.write(row)
                f.write("\n")
