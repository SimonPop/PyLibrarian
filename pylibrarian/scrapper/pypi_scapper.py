import requests
import json
import xmlrpc
from time import sleep
from typing import List
from tqdm import tqdm

class PypiScrapper():
    def __init__(self):
        self.output_file = "pypi_packages.csv"
        self.sleep = 0.5
        self.packages = self.get_packages()

    def get_packages(self) -> List[str]:
        # client = xmlrpc.client.ServerProxy('http://pypi.python.org/pypi')
        # packages = client.list_packages()

        with open("packages.txt", "r") as f:
            packages = f.read().splitlines()

        with open(self.output_file, "r") as f:
            done_packages = [l.split(';')[0] for l in f.readlines()]

        return [p for p in packages if p not in done_packages]

    def get_requirements(self, package_name: str) -> List[str]:
        request = requests.get(f"https://pypi.org/pypi/{package_name}/json")
        if request.status_code != 200:
            print(f"Request failed for {package_name}", request.status_code)
            return []
        else:
            content = request.content
            requires_dist = json.loads(content)['info']['requires_dist']
            if requires_dist is None:
                requires_dist = []
            return [r.split(' ')[0] for r in requires_dist]

    def scrap(self):
        for package in tqdm(self.packages, desc="Parsing packages."):
            requirements = self.get_requirements(package)
            self.write_row(package, requirements)
            sleep(self.sleep)

    def write_row(self, package: str, requirement: List[str]):
        row = f"{package};{requirement}"
        with open(self.output_file, "a") as f:
            f.write(
                row
            )
            f.write("\n")

if __name__ == "__main__":
    pypi_scrapper = PypiScrapper()
    pypi_scrapper.scrap()
    
