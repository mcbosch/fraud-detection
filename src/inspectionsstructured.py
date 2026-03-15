from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.graphstructure import GraphFraudData

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


# ======== Intermediate Strategies =====================

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


class NodeFeaturesByClassStrategy(GraphEDAStrategy):
    """
    Plots KDE distributions of the top-N node features, split by class label.
    Requires node_features to share the same node-ID index as data.classes.
    """

    def __init__(self, top_n: int = 9):
        self.top_n = top_n

    def analyze(self, data: GraphFraudData) -> None:
        id_col   = data.classes.columns[0]
        cls_col  = data.classes.columns[1]

        # Attach class labels to node features by positional alignment
        df = data.node_features.copy().reset_index(drop=True)
        cls = data.classes[[id_col, cls_col]].reset_index(drop=True)
        df['__class__'] = cls[cls_col].values

        # Drop unknown / unlabeled nodes if class is stored as string 'unknown'
        df = df[df['__class__'].astype(str) != 'unknown']

        numeric_cols = [c for c in df.select_dtypes(include='number').columns
                        if c != '__class__'][:self.top_n]

        n = len(numeric_cols)
        ncols = 3
        nrows = (n + ncols - 1) // ncols

        _, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3))
        axes = axes.flatten()
        palette = sns.color_palette("Set2", df['__class__'].nunique())

        for i, col in enumerate(numeric_cols):
            for j, (label, grp) in enumerate(df.groupby('__class__')):
                axes[i].hist(grp[col].dropna(), bins=40, alpha=0.5,
                             label=str(label), color=palette[j], density=True)
            axes[i].set_title(f'Feature {col}', fontsize=9)
            axes[i].legend(fontsize=7)

        for k in range(i + 1, len(axes)):
            axes[k].set_visible(False)

        plt.suptitle('Node feature distributions by class', y=1.01)
        plt.tight_layout()
        plt.show()


# ======== Advanced Strategies =====================

class ConnectedComponentsStrategy(GraphEDAStrategy):
    """
    Analyses connected components of the (undirected) graph:
    number of components, size distribution, and fraction of nodes
    in the largest component.
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

        # Plot: component size distribution (excluding the giant component for readability)
        tail_sizes = sizes[1:] if len(sizes) > 1 else sizes
        plt.figure(figsize=(8, 4))
        plt.hist(tail_sizes, bins=40, color='steelblue', edgecolor='white', log=True)
        plt.title('Component size distribution (excl. giant component)')
        plt.xlabel('Component size')
        plt.ylabel('Count (log)')
        plt.tight_layout()
        plt.show()


class HomophilyStrategy(GraphEDAStrategy):
    """
    Computes edge homophily: the fraction of edges that connect two nodes
    of the same class.  High homophily means fraud nodes tend to connect
    to other fraud nodes.

    H = |{(u,v) ∈ E : class(u) == class(v)}| / |labeled edges|
    """

    def analyze(self, data: GraphFraudData) -> None:
        cmap = _class_map(data)
        edge_cols = data.edges.columns.tolist()
        src_col, dst_col = edge_cols[0], edge_cols[1]

        total, same_class = 0, 0
        class_pair_counts: dict[tuple, int] = {}

        for _, row in data.edges.iterrows():
            u, v = row[src_col], row[dst_col]
            cu, cv = cmap.get(u), cmap.get(v)
            if cu is None or cv is None:
                continue
            total += 1
            if cu == cv:
                same_class += 1
            pair = tuple(sorted([str(cu), str(cv)]))
            class_pair_counts[pair] = class_pair_counts.get(pair, 0) + 1

        homophily = same_class / total if total else float('nan')

        print("= Edge Homophily\t" + "=" * 40)
        print(f"  Labeled edges   : {total:,}")
        print(f"  Same-class edges: {same_class:,}")
        print(f"  Homophily index : {homophily:.4f}")
        print("\n  Edge class-pair counts:")
        for pair, cnt in sorted(class_pair_counts.items(), key=lambda x: -x[1]):
            print(f"    {pair[0]} -- {pair[1]}: {cnt:,}  ({cnt/total*100:.1f}%)")

        # Bar chart of class-pair frequencies
        labels = [f"{p[0]}–{p[1]}" for p in sorted(class_pair_counts)]
        values = [class_pair_counts[p] for p in sorted(class_pair_counts)]
        plt.figure(figsize=(7, 4))
        plt.bar(labels, values, color=sns.color_palette("Set2", len(labels)))
        plt.title(f'Edge class-pair distribution  (homophily = {homophily:.3f})')
        plt.ylabel('Edge count')
        plt.tight_layout()
        plt.show()


class HubNodesStrategy(GraphEDAStrategy):
    """
    Identifies the top-k highest-degree nodes and shows their class breakdown.
    Useful for spotting whether hubs are disproportionately fraudulent.
    """

    def __init__(self, top_k: int = 20):
        self.top_k = top_k

    def analyze(self, data: GraphFraudData) -> None:
        G = _build_graph(data, directed=True)
        cmap = _class_map(data)
        cls_col = data.classes.columns[1]

        degree_df = pd.DataFrame({
            'node': list(G.nodes()),
            'in_degree':  [G.in_degree(n)  for n in G.nodes()],
            'out_degree': [G.out_degree(n) for n in G.nodes()],
        })
        degree_df['total_degree'] = degree_df['in_degree'] + degree_df['out_degree']
        degree_df[cls_col] = degree_df['node'].map(cmap)
        top = degree_df.nlargest(self.top_k, 'total_degree').reset_index(drop=True)

        print(f"= Top-{self.top_k} Hub Nodes\t" + "=" * 40)
        print(top.to_string(index=False))

        # Class breakdown among hubs
        hub_class_counts = top[cls_col].value_counts()
        print(f"\n  Class breakdown among top-{self.top_k} hubs:")
        print(hub_class_counts.to_string())

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        colors = sns.color_palette("Set2", len(hub_class_counts))

        top.set_index('node')[['in_degree', 'out_degree']].plot(
            kind='bar', stacked=True, ax=axes[0], color=['steelblue', 'tomato'])
        axes[0].set_title(f'Top-{self.top_k} hubs: in vs out degree')
        axes[0].set_xlabel('Node ID')
        axes[0].tick_params(axis='x', rotation=45)

        hub_class_counts.plot(kind='bar', ax=axes[1], color=colors)
        axes[1].set_title(f'Class distribution in top-{self.top_k} hubs')
        axes[1].set_ylabel('Count')
        axes[1].tick_params(axis='x', rotation=0)

        plt.tight_layout()
        plt.show()


class TemporalPatternStrategy(GraphEDAStrategy):
    """
    Plots class distribution over time steps, assuming the first
    node-feature column (time_col index) encodes a discrete time step.

    Useful for datasets like Elliptic where feature column 0 is the
    time step (1–49).
    """

    def __init__(self, time_col: int = 0):
        self.time_col = time_col

    def analyze(self, data: GraphFraudData) -> None:
        cls_col = data.classes.columns[1]
        id_col  = data.classes.columns[0]

        # Align node features with class labels by position
        df = data.node_features.copy().reset_index(drop=True)
        time_series = df.iloc[:, self.time_col].rename('time_step')
        cls = data.classes[[id_col, cls_col]].reset_index(drop=True)

        merged = pd.DataFrame({
            'time_step': time_series,
            cls_col: cls[cls_col].values,
        })
        merged = merged[merged[cls_col].astype(str) != 'unknown']

        counts = merged.groupby(['time_step', cls_col]).size().unstack(fill_value=0)
        pct    = counts.div(counts.sum(axis=1), axis=0) * 100

        print("= Temporal Class Patterns\t" + "=" * 30)
        print(counts.head(10))

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        counts.plot(ax=axes[0], marker='o', markersize=3)
        axes[0].set_title('Node count per class over time')
        axes[0].set_ylabel('Node count')
        axes[0].legend(title=cls_col)

        pct.plot(ax=axes[1], marker='o', markersize=3)
        axes[1].set_title('Class fraction over time')
        axes[1].set_ylabel('Fraction (%)')
        axes[1].set_xlabel('Time step')
        axes[1].legend(title=cls_col)

        plt.tight_layout()
        plt.show()


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

    inspector = GraphInspector(GraphSummaryStrategy())
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