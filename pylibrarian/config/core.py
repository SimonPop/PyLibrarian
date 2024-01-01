from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import yaml

class Database(BaseModel):
    google_project: Optional[str] = None
    file: Optional[str] = None

class Config(BaseModel):
    parser: Optional[Database] = None

def create_cfg(file_name: str = None) -> Config:
    with open(file_name, 'r') as stream:
        config = Config(**yaml.safe_load(stream))
    return config

CONFIG = create_cfg(Path(__file__).parent / "config.yaml")