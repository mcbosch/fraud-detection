from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

r"""
EDA methods module following a Strategy Design Pattern.
Follow the EDAStrategy template to add new analysis strategies.
"""


class EDAStrategy(ABC):  # Template
    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> None:
        r"""
        This strategy runs an algorithm and prints/pops the output.

        Parameters
        ----------
            df: pd.DataFrame (data where we want to apply the strategy)
        
        Returns
        -------
            None (prints a result)
        """
        pass

 
# ======== Concrete Strategies =====================

class SummaryStatsStrategy(EDAStrategy):
    def analyze(self, df: pd.DataFrame) -> None:
        print("= Shape\t" + "="*40 )
        print(df.shape)
        print("\n= Data types\t" + "="*40)
        print(df.dtypes)
        print("\n= Descriptive statistics\t" + "=" * 20)
        print(df.describe())
        


class MissingValuesStrategy(EDAStrategy):
    def analyze(self, df: pd.DataFrame) -> None:
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        report = pd.DataFrame({'missing': missing, 'pct': missing_pct})
        report = report[report['missing'] > 0].sort_values('missing', ascending=False)

        print("= Missing values\t" + "="*20)
        if report.empty:
            print("No missing values found.")
        else:
            print(report)


class ClassImbalanceStrategy(EDAStrategy):
    def __init__(self, target_col: str = 'Class'):
        self.target_col = target_col

    def analyze(self, df: pd.DataFrame) -> None:
        counts = df[self.target_col].value_counts()
        pct = (counts / len(df) * 100).round(2)

        print(f"= Class distribution ({self.target_col})\t" + "="*20)
        print(pd.DataFrame({'count': counts, 'pct': pct}))

        _, axes = plt.subplots(1, 2, figsize=(10, 4))
        counts.plot(kind='bar', ax=axes[0], color=['steelblue', 'tomato'])
        axes[0].set_title('Class counts')
        axes[0].set_xlabel(self.target_col)
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=0)

        axes[1].pie(counts, labels=counts.index, autopct='%1.2f%%',
                    colors=['steelblue', 'tomato'], startangle=90)
        axes[1].set_title('Class proportions')

        plt.tight_layout()
        plt.show()


class FeatureDistributionStrategy(EDAStrategy):
    def __init__(self, max_features: int = 20, target_col: str = 'Class'):
        self.max_features = max_features
        self.target_col = target_col

    def analyze(self, df: pd.DataFrame) -> None:
        numeric_cols = [
            c for c in df.select_dtypes(include='number').columns
            if c != self.target_col
        ][:self.max_features]

        n = len(numeric_cols)
        ncols = 4
        nrows = (n + ncols - 1) // ncols

        _, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            axes[i].hist(df[col].dropna(), bins=50, color='steelblue', edgecolor='white')
            axes[i].set_title(col, fontsize=9)
            axes[i].set_xlabel('')

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle('Feature distributions', y=1.01)
        plt.tight_layout()
        plt.show()


class CorrelationStrategy(EDAStrategy):
    def __init__(self, target_col: str = 'Class', top_n: int = 15):
        self.target_col = target_col
        self.top_n = top_n

    def analyze(self, df: pd.DataFrame) -> None:
        corr = df.select_dtypes(include='number').corr()

        # Top correlated features with the target
        if self.target_col in corr.columns:
            top_cols = (
                corr[self.target_col]
                .abs()
                .sort_values(ascending=False)
                .head(self.top_n + 1)
                .index.tolist()
            )
            corr = corr.loc[top_cols, top_cols]

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=False, cmap='coolwarm', center=0,
                    linewidths=0.5, square=True, vmax=1, vmin=-1)
        plt.title('Correlation matrix')
        plt.tight_layout()
        plt.show()


########################################################## 

class DataInspector:
    def __init__(self, strategy: EDAStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: EDAStrategy) -> None:
        self._strategy = strategy

    def run(self, df: pd.DataFrame) -> None:
        self._strategy.analyze(df)

    def run_all(self, df: pd.DataFrame, strategies: list[EDAStrategy]) -> None:
        for strategy in strategies:
            self.set_strategy(strategy)
            self.run(df)


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from data.dataloader import DataLoaderFactory

    # df = DataLoaderFactory.create('zip').load_data('data/raw/creditcard.csv.zip')

    # inspector = DataInspector(SummaryStatsStrategy())

    # inspector.run_all(df, [
    #     SummaryStatsStrategy(),
    #     MissingValuesStrategy(),
    #     ClassImbalanceStrategy(target_col='Class'),
    #     FeatureDistributionStrategy(target_col='Class'),
    #     CorrelationStrategy(target_col='Class'),
    # ])
    pass
