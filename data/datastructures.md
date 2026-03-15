# Fraud Detection Datasets

Note: Documentet created with `claude.code` to be knowledgeable of which datasets I can extend the project.

## Current Datasets

**Credit Card Fraud Detection** (MLG-ULB / Kaggle)
- Format: Tabular CSV — 284,807 transactions × 31 columns
- Columns: `Time`, `V1–V28` (PCA-anonymized), `Amount`, `Class`
- Labels: 492 fraudulent (~0.17%), 284,315 legitimate
- **Not graph-structured.** Each row is an independent transaction with no relational links.

---

## Graph-Structured Fraud Datasets

### Natively Graph-Structured

| Dataset | Nodes | Edges | Size | Source |
|---|---|---|---|---|
| **Elliptic Bitcoin** | Bitcoin transactions | Payment flows between transactions | 203k nodes, 234k edges, 166 features | [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) |
| **Elliptic++** | Transactions + 822k wallet addresses | Multiple graph types (tx→tx, addr→addr, etc.) | Superset of Elliptic | [GitHub](https://github.com/git-disl/EllipticPlusPlus) |
| **DGraph** | Loan platform users | Emergency contact relationships (real social graph) | 3.7M nodes, 4M edges, 17 features | [dgraph.xinye.com](https://dgraph.xinye.com) |
| **T-Finance** | Financial accounts | Transaction relationships | Thousands of nodes, 10 features | [GitHub](https://github.com/squareRoot3/Rethinking-Anomaly-Detection) / [Kaggle](https://www.kaggle.com/datasets/andrewtaj/tsocial-tfinance) |
| **T-Social** | Social media accounts | Persistent friendships (>3 months) | ~5M nodes, ~100M edges | [GitHub](https://github.com/squareRoot3/Rethinking-Anomaly-Detection) / [Kaggle](https://www.kaggle.com/datasets/andrewtaj/tsocial-tfinance) |

### Tabular but Graph-Interpretable (Graph Construction Required)

| Dataset | Natural Graph Interpretation | Fraud Rate | Source |
|---|---|---|---|
| **YelpChi** | Reviews as nodes; edges = shared user / same rating+month / TF-IDF text similarity | ~14.5% | DGL built-in: `dgl.data.FraudYelpDataset` |
| **Amazon Review Fraud** | Reviewers as nodes; edges = shared product / same star+week / similar review text | ~9.5% | DGL built-in: `dgl.data.FraudAmazonDataset` |
| **IEEE-CIS** | Transactions + entity nodes (card, device, IP, email); edges = shared identifiers | ~3.5% | [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data) |
| **PaySim** | Accounts as nodes, transactions as directed edges | ~0.13% | [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) |
| **IBM AML (NeurIPS 2023)** | Bank accounts as nodes, transactions as directed edges; designed for GNN use | varies (HI/LI splits) | [Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) / [GitHub](https://github.com/IBM/AML-Data) |

---

## Notes

- **Best GNN benchmarks to start with:** YelpChi and Amazon — most common in academic GNN fraud papers and available directly via DGL with no data wrangling.
- **Most realistic graph structure:** Elliptic Bitcoin — graph is inherent to the blockchain UTXO model, not constructed post-hoc.
- **Largest scale:** T-Social (~100M edges) and IBM AML (~175M transactions in the large variant).
- **Key distinction:** Natively graph datasets reflect real-world network topology. Constructed graphs depend on edge-definition choices (shared attributes, similarity thresholds) which can significantly affect benchmark results and must be reported explicitly in papers.

### References
- ICML 2022 — T-Finance/T-Social: ["Rethinking Graph Neural Networks for Anomaly Detection"](https://github.com/squareRoot3/Rethinking-Anomaly-Detection)
- NeurIPS 2022 — DGraph: ["A Large-Scale Financial Dataset for Graph Anomaly Detection"](https://arxiv.org/abs/2207.03579)
- NeurIPS 2023 — IBM AML: ["Realistic Synthetic Financial Transactions for Anti-Money Laundering"](https://proceedings.neurips.cc/paper_files/paper/2023/file/5f38404edff6f3f642d6fa5892479c42-Paper-Datasets_and_Benchmarks.pdf)
- CIKM 2020 — YelpChi: ["Enhancing Graph Neural Network-based Fraud Detection via Imbalanced Graph Learning"](https://paperswithcode.com/dataset/yelpchi)
- Elliptic: ["Anti-Money Laundering in Bitcoin"](https://arxiv.org/pdf/1908.02591)
