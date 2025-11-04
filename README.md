# Unified Latent Space

## Overview
Conference on Test Security (COTS) 2025 research on a Unified Latent Space IRT framework integrating:
- Response accuracy
- Response time
- Process / interaction sequences
  
**Focus:** Relationship between latent-space influence (gamma) and local item dependence.

## Key Goals
1. Specify LS-IRT variants for separate data modalities.
2. Qualify and quantify item dependence per modality.
3. Introduce and evaluate a unified gamma parameter across modalities.
4. Validate framework on small empirical dataset.

## Repository Structure (high-level)
```
unified-LS/
  analysis/                 # Stan models & run scripts
    models/                 # Stan model files (*.stan)
    run_ls_models.py        # Unified model script
  exploratory/              # Scratch / exploratory notebooks
  utils/                    # Standalone utilities
    data_import.py
    rotate.py
  doc_src/                  # Sphinx documentation source
  docs/                     # Built HTML docs (GitHub Pages)
  data/                     # Placeholder for raw / processed data
  results/                  # Saved model outputs (parquet, csv)
  assets/                   # Figures / static assets
  presentation/             # Slide / poster materials
  .github/                  # CI workflows
  .vscode/                  # Editor settings / snippets
  .poetry/                  # Poetry plugin config
  README.md
  pyproject.toml
  requirements.txt
  requirements-exact.txt
  environment.yml
  LICENSE
```

## Installation

Clone the repository:
```bash
git clone https://github.com/Wmuntean/unified-LS.git
cd unified-LS
```

Set up the Python environment (choose one):

**Using pip:**
```bash
pip install -r requirements.txt
```

**Or with Poetry:**
```bash
poetry install
```

````{note}
This is a research repository, not a pip-installable package. Scripts and modules are run directly from the source tree.

Set the import path for development:
```python
import sys
sys.path.insert(0, "path/to/unified_ls")
import unified_ls
```
````

## Usage
- **Run Analyses:** 
```bash
python utils/data_import.py
python analysis/run_ls_models.py
code analysis/figures.ipynb    # Opens notebook in VS Code
```

## Modeling Framework
Conceptual layers:
1. Item parameter layer (difficulty, time intensity, interaction demands, etc.)
2. Person latent trait layer (abilities, speededness, engagement)
3. Latent space mapping person and item interactions and defining latent distances
4. Gamma parameter modulating cross-modality influence
5. Posterior inclusion probability of latent space

## Data
- **Data Sources:** Raw data is located in local folders and network copies.
- **Data Structure:** Data are pre-processed the `data_import.py` script.


## Modules
<!-- data_import -->
[`data_import.py`](#utils.data_import):
- Imports, parses, and cleans item-level interaction and response data from XML and zipped files, producing a unified DataFrame for modeling and analyses.
<!-- rotate -->
[`rotate.py`](#utils.rotate): 
- Provides functions for aligning latent space coordinates across MCMC chains using Procrustes analysis, including extraction and replacement of latent coordinates in Stan draws.
<!-- run_ls_models -->
[`run_ls_models.py`](#analysis.run_ls_models): 
- Executes Stan latent space models (IRT, RT, process, unified) on preprocessed data, manages model fitting, result saving, and latent space alignment. Supports multiple model types and outputs aligned parameter draws for downstream analysis.
<!-- end modules -->

## Analysis
<!-- figures -->
[`figures.ipynb`](#../_notebooks/figures):
- Generates descriptive and comparative plots for latent space models, including process counts distributions, residual relationships, and latent space alignment visualizations.  
- Loads model outputs, computes summary statistics, and visualizes relationships between modalities (accuracy, response time, process counts) using stem plots, regression plots, and Procrustes-aligned barbell plots.  
- Outputs publication-ready figures to the `assets/` directory for downstream analysis and reporting.
<!-- end analysis -->

````{note}
API Documentation is available at 

Set the import path for development:
```python
import sys
sys.path.insert(0, "path/to/unified_ls")
import unified_ls
```
````

## Documentation
API documentation is available at [https://wmuntean.github.io/unified-LS/](https://wmuntean.github.io/unified-LS/index_documentation.html)

## Contributors
- [@wmuntean](https://github.com/Wmuntean)

## Citation
Presented at the Conference on Test Security (COTS), UMass Amherst, 2025.  


```bibtex
@inproceedings{muntean2025unifiedLS,
  author       = {William Muntean and Zhuoran Wang and Joe Betts},
  title        = {A Unified Latent Space IRT Framework Integrating Accuracy, Response Time, and Process Data},
  booktitle    = {Conference on Test Security (COTS)},
  year         = {2025},
  address      = {Amherst, MA, USA},
  organization = {University of Massachusetts Amherst},
  url          = {https://github.com/Wmuntean/unified-LS}
}
```

## License and IP Notice

This research and its contents are proprietary and contain intellectual property owned by William Muntean. Unauthorized use, copying, distribution, or disclosure of any part of this work is strictly prohibited without prior written consent. For inquiries regarding usage rights, please contact williamjmuntean@gmail.com.
