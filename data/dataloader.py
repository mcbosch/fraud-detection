from abc import ABC, abstractmethod
import os
import zipfile

import pandas as pd

r"""
This script builds a data loader following a Factory Design Pattern.
Follow the templates to add new Loader classes.
"""

class DataLoader(ABC): # Template
    @abstractmethod
    def load_data(self, path: str) -> pd.DataFrame:
        pass


class ZipLoader(DataLoader):
    def load_data(self, path: str) -> pd.DataFrame:
        with zipfile.ZipFile(path, 'r') as zf:
            csv_files = [name for name in zf.namelist() if name.endswith('.csv')]
            if not csv_files:
                raise ValueError(f"No CSV files found inside '{path}'.")
            with zf.open(csv_files[0]) as f:
                return pd.read_csv(f)


class KaggleLoader(DataLoader):
    _datasets = {
        'ccfraud-kaggle': 'mlg-ulb/creditcardfraud',
    }

    def load_data(self, path: str) -> pd.DataFrame:
        if path not in self._datasets:
            raise ValueError(
                f"Dataset '{path}' is not supported. "
                f"Available: {list(self._datasets.keys())}"
            )
        import kaggle  # type: ignore[import-untyped]

        slug = self._datasets[path]
        output_dir = os.path.join('data', 'raw')
        os.makedirs(output_dir, exist_ok=True)

        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(slug, path=output_dir, quiet=False)

        zip_path = os.path.join(output_dir, slug.split('/')[-1] + '.zip')
        return ZipLoader().load_data(zip_path)


class DataLoaderFactory:
    _registry = {
        'zip': ZipLoader,
        'kaggle': KaggleLoader,
    }

    @staticmethod
    def create(source_type: str) -> DataLoader:
        loader_class = DataLoaderFactory._registry.get(source_type)
        if loader_class is None:
            raise ValueError(
                f"Unknown source type: '{source_type}'. "
                f"Available: {list(DataLoaderFactory._registry.keys())}"
            )
        return loader_class()


if __name__ == '__main__':
    loader = DataLoaderFactory.create('zip')
    df = loader.load_data('data/raw/creditcard.csv.zip')
    print(df.shape)
    print(df.head())
