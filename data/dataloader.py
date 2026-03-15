from abc import ABC, abstractmethod
from data.graphstructure import GraphFraudData

import os
import zipfile

import pandas as pd

r"""
This script builds a data loader following a Factory Design Pattern.
Follow the templates to add new Loader classes.
"""

class DataLoader(ABC): # Template
    @abstractmethod
    def load_data(self, path: str, graph_structured = False) -> pd.DataFrame | GraphFraudData:
        r"""
        Method that loads the data and returns a python object designed to store the data. 
        NOTE: for graph_structured data we should have 3 files:
            1. node_features: a .csv file with the features of each node. Each row is a node.
            2. edges: a .csv file with at least two columns, tail and head of the edge. 
            3. classes: class of each node.

        Parameters
        ----------
            . path: path of the documents storing data
            . graph_structured: indicates if our data is previously treated as a graph. 
        Returns
        -------
            If graph_structured = True returns a GraphFraudData object.
            If graph_structured = False returns a pd.DataFrame object.
        """
        pass


class ZipLoader(DataLoader):
    
    def load_data(self, path: str, graph_structured = False) -> pd.DataFrame | GraphFraudData:
        
        if not graph_structured: 
            with zipfile.ZipFile(path, 'r') as zf:
                csv_files = [name for name in zf.namelist() if name.endswith('.csv')]
                if not csv_files:
                    raise ValueError(f"No CSV files found inside '{path}'.")
                with zf.open(csv_files[0]) as f:
                    return pd.read_csv(f)

        with zipfile.ZipFile(path, 'r') as zf:
            csv_files = [name for name in zf.namelist() if name.endswith('.csv')]
            if not csv_files:
                raise ValueError(f"No CSV files found inside '{path}'.")
            if not len(csv_files) >= 3:
                raise ValueError(
                    f"Graph Structured data needs 3 csv files"
                    f"\n\t-node features"
                    f"\n\t-edge features"
                    f"\n\t-class of nodes"
                    f"\n'{path}' only has {len(csv_files)} CSV files:"
                    f"\n{csv_files}"
                    )
            
            features_read, edges_read, classes_read = False, False, False
            for csv in csv_files:
                if 'features' in csv:
                    with zf.open(csv) as f:
                        data_features = pd.read_csv(f, header=None) # Header none because is Anonymazed data
                        features_read = True
                if 'edge' in csv:
                    with zf.open(csv) as f:
                        data_edges = pd.read_csv(f)
                        edges_read = True
                if 'class' in csv:
                    with zf.open(csv) as f:
                        data_classes = pd.read_csv(f)
                        classes_read = True
            
            
            if not features_read or not edges_read or not classes_read :
                raise ValueError(
                    f"There is missinng data in '{path}':"
                    f"\n\tNode Features Data: {features_read}"
                    f"\n\tEdges Data: {edges_read}"
                    f"\n\tClasses Data: {classes_read}"
                )
            
            return GraphFraudData(data_features, data_edges, data_classes)
                    
class KaggleLoader(DataLoader):
    _datasets = {
        'ccfraud-kaggle': 'mlg-ulb/creditcardfraud',
        'elliptic-kaggle': 'ellipticco/elliptic-data-set',
    }

    def load_data(self, path: str, graph_structured = False) -> pd.DataFrame | GraphFraudData:
        if path not in self._datasets:
            raise ValueError(
                f"\nDataset '{path}' is not supported. "
                f"\nAvailable: {list(self._datasets.keys())}"
            )
        import kaggle  # type: ignore[import-untyped]

        slug = self._datasets[path]
        output_dir = os.path.join('data', 'raw')
        os.makedirs(output_dir, exist_ok=True)

        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(slug, path=output_dir, quiet=False)

        zip_path = os.path.join(output_dir, slug.split('/')[-1] + '.zip')
        return ZipLoader().load_data(zip_path, graph_structured = graph_structured)


class DataLoaderFactory:
    _registry = {
        'zip': ZipLoader,
        'kaggle': KaggleLoader,
    }

    @staticmethod
    def create(source_type: str) -> DataLoader | GraphFraudData:
        loader_class = DataLoaderFactory._registry.get(source_type)
        if loader_class is None:
            raise ValueError(
                f"Unknown source type: '{source_type}'. "
                f"Available: {list(DataLoaderFactory._registry.keys())}"
            )
        return loader_class()


if __name__ == '__main__':
    loader = DataLoaderFactory.create('zip')
    df = loader.load_data('data/raw/elliptic-data-set.zip', graph_structured=True)
    print(df.head())
