# unified-LS <!-- omit in toc -->

## Overview
- **Project Description:**
    - COTS 2025 Research on Unified Latent Space IRT Model
    - Key focuses:
        - List any important focus areas.
- **Research Goals:**
    1. Goal 1
    2. Goal 2
    3. Goal 3
- **Scope:**
    - Define the scope of the research.
- **Note:**
    - List any important notes.
    - Example: The repository does not contain any data nor results. The raw data references local and network copies.
## Table of Contents
1. [Overview](#overview)
2. [Table of Contents](#table-of-contents)
3. [Installation](#installation)
    1. [Clone repository:](#clone-repository)
    2. [Install requirements in working environment:](#install-requirements-in-working-environment)
    3. [Set system path in python editor:](#set-system-path-in-python-editor)
4. [Documentation](#documentation)
5. [Data](#data)
6. [Usage](#usage)
7. [Modules](#modules)
8. [Analysis](#analysis)
9. [Results](#results)
10. [Contributors](#contributors)
11. [License and IP Notice](#license-and-ip-notice)

## Installation

### Clone repository:

Clone the repository:
```bash
git clone https://github.com/Wmuntean/unified-LS.git
cd unified-LS
```

### Install requirements in working environment:

```bash
pip install -r requirements.txt
```
or
```bash
poetry install
```
### Set system path in python editor:

```python
import sys
sys.path.insert(0, R"path\to\unified_ls")

# Confirm development version
import unified_ls
print(unified_ls.__version__)
# Should output: {version}+
```


## Documentation


- API documentation is available at https://Wmuntean.github.io/unified_ls/


## Data
- **Data Sources:** Raw data is located in local folders and network copies.
- **Data Structure:** Data are processed via ... See `data_import.py` for details.
  
## Usage
- **Run Analyses:** 
```bash
python utils/data_import.py
python analysis/module_name_1.py
python analysis/module_name_2.py
```

## Modules
<!-- data_import -->
[`data_import.py`](#utils.data_import):
- Data import description.
<!-- module_name 1 -->
[`module_name.py`](#module.location): 
- Module description.
<!-- module_name 2 -->
[`module_name.py`](#module.location): 
- Module description.
<!-- end modules -->

## Analysis
<!-- analysis 1 -->
[`analysis_notebook.ipynb`](#../_collections/notebooks/analysis_notebook):
- Analysis description.
```{attention}
- Some attention admonition. Others include:
    - attention
    - caution
    - danger
    - error
    - hint
    - important
    - note
    - tip
    - warning
    - admonition (with custom title)
    - seealso
```
<!-- analysis 2 -->
[`analysis_script.py`](#analysis.analysis_script):
- Analysis description.
- MLflow results can be found [here](http_link_to_databricks_mflow).
<!-- end analysis -->

## Results
- **Key Findings:**
    - Results description

## Contributors
- [@wmuntean](https://github.com/Wmuntean)

## License and IP Notice
This research and its contents are proprietary and contain intellectual property owned by William Muntean. Unauthorized use, copying, distribution, or disclosure of any part of this work is strictly prohibited without prior written consent. For inquiries regarding usage rights, please contact williamjmuntean@gmail.com.
****