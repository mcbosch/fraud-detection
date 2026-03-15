from dataclasses import dataclass

import pandas as pd

@dataclass
class GraphFraudData:
    r"""
    This class is a dataclass object to work with datasets that are graph-structured.
    It has 3 attributes:
        - node_features: pd.DataFrame with features of nodes
        - edges: pd.DataFrame with edges (at least two variables, tail and head of the edge)
        - classes: pd.DataFrame indicating class of each node. 
    """
    node_features: pd.DataFrame
    edges: pd.DataFrame
    classes: pd.DataFrame

    def head(self, n = 5):
        # Prints head of each dataframe
        print("__ Node Features " + "_"*20)
        print(f"Shape: \t",self.node_features.shape)
        print(self.node_features.head(n))
        print("\n__ Edges " + "_"*28)
        print(f"Shape: \t",self.edges.shape)
        print(self.edges.head(n))
        print("\n__ Node Classes " + "_"*21)
        print(f"Shape: \t",self.classes.shape)
        print(self.classes.head(n))