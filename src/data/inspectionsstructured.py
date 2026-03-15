from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data.graphstructure import GraphFraudData

r"""
EDA methods for graph-structured fraud data following a Strategy Design Pattern.
Strategies range from basic summaries to advanced graph-theoretic analyses.
Follow the GraphEDAStrategy template to add new strategies.
"""


class GraphEDAStrategy(ABC):  # Template
    @abstractmethod
    def analyze(self, data: GraphFraudData) -> None:
        r"""
        Runs an analysis and prints / plots the output.

        Parameters
        ----------
            data: GraphFraudData
        Returns
        -------
            None (prints / shows a result)
        """
        pass


# ======== Helpers =====================

def _build_graph(data: GraphFraudData, directed: bool = True) -> nx.DiGraph | nx.Graph:
    """Build a NetworkX graph from the GraphFraudData edges."""
    edge_cols = data.edges.columns.tolist()
    src, dst = edge_cols[0], edge_cols[1]
    G = nx.from_pandas_edgelist(
        data.edges, source=src, target=dst,
        create_using=nx.DiGraph() if directed else nx.Graph()
    )
    return G


def _class_map(data: GraphFraudData) -> dict:
    """Return {node_id: class_label} from data.classes (first two columns)."""
    cols = data.classes.columns.tolist()
    return dict(zip(data.classes[cols[0]], data.classes[cols[1]]))


# ======== Basic Strategies =====================

class GraphSummaryStrategy(GraphEDAStrategy):
    """Prints node count, edge count, feature dimensionality, and class count."""

    def analyze(self, data: GraphFraudData) -> None:
        n_nodes = len(data.node_features)
        n_edges = len(data.edges)
        n_features = data.node_features.shape[1]
        classes = data.classes['class'].value_counts().to_dict()
        n_classes = len(classes)
        class_labels = list(classes.keys())

        print("= Graph Summary\t" + "=" * 40)
        print(f"  Nodes             : {n_nodes:,}")
        print(f"  Edges             : {n_edges:,}")
        print(f"  Node features     : {n_features}")
        print(f"  Unique classes    : {n_classes}  {class_labels}")
        print(fr"""  Proportion classes: 
                        {class_labels[0]}: {round(100*classes[class_labels[0]]/n_nodes,3)} %
                        {class_labels[1]}: {round(100*classes[class_labels[1]]/n_nodes,3)} %
                        {class_labels[2]}: {round(100*classes[class_labels[2]]/n_nodes,3)} %""")
        print(f"\n  Avg degree (undirected) : {2 * n_edges / n_nodes:.2f}" if n_nodes else "")

        print("\n-- Node features (head) --")
        print(data.node_features.head())
        print("\n-- Edges (head) --")
        print(data.edges.head())
        print("\n-- Classes (head) --")
        print(data.classes.head())


class NodeMissingValuesStrategy(GraphEDAStrategy):
    """Reports missing values in node features."""

    def analyze(self, data: GraphFraudData) -> None:
        df = data.node_features
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        report = pd.DataFrame({'missing': missing, 'pct': missing_pct})
        report = report[report['missing'] > 0].sort_values('missing', ascending=False)

        print("= Node Feature Missing Values\t" + "=" * 20)
        if report.empty:
            print("No missing values found.")
        else:
            print(report)


class ClassDistributionStrategy(GraphEDAStrategy):
    """Bar + pie chart of class distribution across nodes."""

    def analyze(self, data: GraphFraudData) -> None:
        class_col = data.classes.columns[1]
        counts = data.classes[class_col].value_counts().sort_index()
        pct = (counts / counts.sum() * 100).round(2)

        print(f"= Class Distribution ({class_col})\t" + "=" * 20)
        print(pd.DataFrame({'count': counts, 'pct': pct}))

        _, axes = plt.subplots(1, 2, figsize=(10, 4))
        colors = sns.color_palette("Set2", len(counts))

        counts.plot(kind='bar', ax=axes[0], color=colors)
        axes[0].set_title('Node class counts')
        axes[0].set_xlabel(class_col)
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=0)

        axes[1].pie(counts, labels=counts.index, autopct='%1.2f%%',
                    colors=colors, startangle=90)
        axes[1].set_title('Node class proportions')

        plt.tight_layout()
        plt.show()


class DegreeDistributionStrategy(GraphEDAStrategy):
    """
    Computes in-degree, out-degree, and total degree for each node
    and plots their distributions.
    """

    def analyze(self, data: GraphFraudData) -> None:
        G = _build_graph(data, directed=True)

        in_deg  = pd.Series(dict(G.in_degree()),  name='in_degree')
        out_deg = pd.Series(dict(G.out_degree()), name='out_degree')
        tot_deg = (in_deg + out_deg).rename('total_degree')

        print("= Degree Distribution\t" + "=" * 40)
        stats = pd.DataFrame({'in_degree': in_deg, 'out_degree': out_deg,
                               'total_degree': tot_deg})
        print(stats.describe().round(2))

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, series, color in zip(axes,
                                      [in_deg, out_deg, tot_deg],
                                      ['steelblue', 'tomato', 'seagreen']):
            ax.hist(series, bins=50, color=color, edgecolor='white', log=True)
            ax.set_title(series.name.replace('_', ' ').title())
            ax.set_xlabel('Degree')
            ax.set_ylabel('Count (log)')

        plt.suptitle('Degree distributions (log scale)')
        plt.tight_layout()
        plt.show()


# ======== Advanced Strategies =====================

class ConnectedComponentsStrategy(GraphEDAStrategy):
    """
    Analyses connected components of the (undirected) graph:
    number of components, size distribution, and fraction of nodes
    in the largest component.

    NOTE: This strategy is a general strategy, is not focused on 
    the Elliptic Data Set where we have previous knowledge on connected 
    components. 

    This Strategy is not recomended for the Elliptic Data Set.
    For Elliptic Data Set see: SubGraphGroupBy
    """

    def analyze(self, data: GraphFraudData) -> None:
        G = _build_graph(data, directed=False)
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        sizes = [len(c) for c in components]
        n_nodes = G.number_of_nodes()

        print("= Connected Components\t" + "=" * 40)
        print(f"  Number of components   : {len(components):,}")
        print(f"  Largest component      : {sizes[0]:,} nodes  ({sizes[0]/n_nodes*100:.1f}%)")
        print(f"  Isolated nodes         : {sum(1 for s in sizes if s == 1):,}")
        print(f"\n  Size stats:")
        print(pd.Series(sizes, name='component_size').describe().round(2))

        # Plot: size of each componsent
        tail_sizes = sizes[1:] if len(sizes) > 1 else sizes
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(components)), sizes, color='steelblue')
        plt.title('Component Size')
        plt.xlabel('Component')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()


class SubGraphGroupBy(GraphEDAStrategy):
    def __init__(self, var: str) -> None:
        self.by = var

    def analyze(self, data: GraphFraudData) -> None:
        pass


##########################################################

class GraphInspector:
    def __init__(self, strategy: GraphEDAStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: GraphEDAStrategy) -> None:
        self._strategy = strategy

    def run(self, data: GraphFraudData) -> None:
        self._strategy.analyze(data)

    def run_all(self, data: GraphFraudData, strategies: list[GraphEDAStrategy]) -> None:
        for strategy in strategies:
            self.set_strategy(strategy)
            self.run(data)


if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from data.dataloader import DataLoaderFactory

    data = DataLoaderFactory.create('zip').load_data(
        'data/raw/elliptic-data-set.zip', graph_structured=True
    )

    inspector = GraphInspector(ConnectedComponentsStrategy())
    inspector.run(data)
    # inspector.run_all(data, [
    #     GraphSummaryStrategy(),
    #     NodeMissingValuesStrategy(),
    #     ClassDistributionStrategy(),
    #     DegreeDistributionStrategy(),
    #     NodeFeaturesByClassStrategy(top_n=9),
    #     ConnectedComponentsStrategy(),
    #     HomophilyStrategy(),
    #     HubNodesStrategy(top_k=20),
    #     TemporalPatternStrategy(time_col=0),
    # ])
    pass