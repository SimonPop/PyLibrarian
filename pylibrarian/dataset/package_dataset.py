import pandas as pd
import ast
import torch
import numpy as np
import jax.numpy as jnp
from typing import Dict
from pylibrarian.recommender.models import AttentionModel

def trim_name(name: str) -> str:
    for delimiter in ['>', '==', '<', '~=', "["]:
        split_name = name.split(delimiter)
        if len(split_name):
            name = split_name[0]
    return name


def pad(l: list, target_length: int, padding_token: int) -> list:
    if len(l) > target_length:
        return l[:target_length]
    else:
        return l + [padding_token]*(target_length - len(l))

class PackageDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame):
        self.token_offset = 1
        self.padding_size = 20
        self.padding_token = 0
        self.df = self.prepare_df(df) 
        self.X, self.y, self.labels = self.prepare_array(np.array(self.df["tokens"])
)
    def prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df["packages"] = df["libs"].apply(ast.literal_eval)
        df["packages"] = df["packages"].apply(lambda ps: [trim_name(p) for p in ps])
        self.tokenizer = self.get_tokenizer(df)
        df["tokens"] = df["packages"].apply(lambda ps: pad([self.tokenizer[p] for p in ps], self.padding_size + 1, self.padding_token))
        df['length'] = df["packages"].apply(lambda p: len(p))
        return df[df["length"] > 3]
    
    def prepare_array(self, array: np.ndarray):
        pool = np.arange(self.token_offset, len(self.tokenizer) + self.token_offset)
        array = np.array([np.random.permutation(r) for r in array])
        positive_y = array[:,0]
        negative_y = [np.random.choice(np.delete(pool, row - self.token_offset), 1)[0] for row in array]
        X = array[:,1:]
        X = np.concatenate((X, X))
        y = np.concatenate((positive_y, negative_y)).reshape(-1,1)
        labels = np.concatenate((np.ones_like(positive_y), np.zeros_like(negative_y)))
        return X, y, labels

    def get_tokenizer(self, df: pd.DataFrame):
        """Creates mapping package -> ID."""
        all_packages = set(
            [x for packages in df['packages'] for x in packages]
        )
        return {package: i+self.token_offset for i, package in enumerate(all_packages)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return {"x": self.X[index], "y":self.y[index], "label": self.labels[index] }
    
    def batched_example(self) -> Dict[str, jnp.array]:
        """Generates a sample of the dataset to instanciate Haiku forward functions.

        Returns:
            Tuple[jnp.array, jnp.array]: X and Y samples.
        """
        return {"x": self[0]['x'][None,:], "y": self[0]['y'][None,:]}