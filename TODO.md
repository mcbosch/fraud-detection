# Project TODO

This file outlines the initial steps and priorities for the fraud detection project. Tasks are ordered with higher priority first and include detailed explanations where necessary.

## High Priority Tasks

1. **Dataset Acquisition and Setup**
   - Visit the [Amazon Science Fraud Dataset Benchmark](https://github.com/amazon-science/fraud-dataset-benchmark) repository.
   - Download the relevant data files and understand the structure and labels.
   - Store raw data under `data/raw/` and create scripts in `src/data` for downloading and validating data integrity.
   - Ensure licensing and citation information is documented.

2. **Environment Initialization**
   - Create a Python virtual environment and list dependencies (e.g., `pandas`, `scikit-learn`, `networkx`, `torch`, `scikit-tda`, `torch-geometric`).
   - Develop a `requirements.txt` or `environment.yml` file.
   - Set up basic project scaffolding (folders enumerated in README).
   - Add linting/formatting configuration (e.g., `flake8`, `black`).

3. **Exploratory Data Analysis (EDA)**
   - Use Jupyter notebooks to perform initial data exploration: class imbalance, feature distributions, missing values.
   - Document findings in `notebooks/eda.ipynb`.
   - Generate baseline plots and summary tables.

4. **Baseline Models**
   - Implement simple models such as logistic regression, decision trees, and random forests.
   - Write training and evaluation scripts under `src/models/`.
   - Track performance metrics (accuracy, precision, recall, F1) and document results.

## Medium Priority Tasks

5. **Feature Engineering**
   - Investigate domain-specific features and transformations.
   - Create pipelines for preprocessing.
   - Store intermediate datasets in `data/processed/`.

6. **Topological Data Analysis (TDA) Pipeline**
   - Research libraries like `scikit-tda` and `gudhi`.
   - Develop modules to compute persistence diagrams and vectorize topological features.
   - Integrate TDA features into model training.

7. **Graph Neural Network Prototyping**
   - Model transactional or entity relationships as graphs.
   - Use `torch-geometric` or `dgl` to build GNN architectures.
   - Experiment with graph construction methods and training routines.

## Lower Priority Tasks

8. **Evaluation & Validation Framework**
   - Implement cross-validation and stratified splits.
   - Add scripts for generating ROC curves, confusion matrices, and other diagnostics.

9. **Testing and CI/CD**
   - Add unit tests for data processing and model components.
   - Integrate with a CI service (GitHub Actions, etc.) to run tests automatically.

10. **Documentation & Reporting**
    - Expand `README.md` with setup/usage instructions.
    - Write technical reports or blog posts summarizing methodology and results.

---

Tasks should be updated regularly as the project evolves. Early focus should be on acquiring the data, setting up the environment, and establishing baseline models to create a foundation for more complex methods.