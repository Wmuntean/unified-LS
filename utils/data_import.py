# ====================================================================
# Author: William Muntean
# Copyright (C) 2025 William Muntean. All rights reserved.
#
# Licensed under the GPL v3 License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://opensource.org/licenses/GPL v3
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ====================================================================

"""
=======================================================
Process Data Import Module
=======================================================

This module provides functionality to parse and process item-level
interaction data from XML files and zipped archives, clean parsed data,
and collapse item scores into no more than three categories with roughly
equal observation counts per category.

Data Processing Workflow
========================

The process is performed in the following stages:

1. **Parsing**:
   - Extracts item-level interaction metrics from XML files.

2. **Cleaning**:
   - Cleans and structures parsed data for modeling.

3. **Batch Processing**:
   - Processes multiple zipped XML files and concatenates results.

4. **Score Collapsing**:
   - Collapses item scores into no more than three categories per item,
   using quantile binning if more than three unique scores exist.

Final DataFrame Variables
=========================
The final DataFrame produced by this module contains the following columns:

- ``person_id``: 1-indexed person identifier.
- ``item_id``: 1-indexed item identifier.
- ``itemset_id``: 1-indexed item set identifier.
- ``rt``: Response time (seconds) for each item.
- ``op_theta``: Operational theta (ability estimate).
- ``item_type``: Item type label.
- ``score``: Item score (integer, collapsed if needed).
- ``total_interactions``: Number of interactions per item.
- ``exhibit_interactions``: Number of exhibit interactions per item.
- ``has_exhibit``: Boolean, whether item has exhibit.
- ``response_selections``: Number of response selections.
- ``response_changes``: Number of response changes.
- ``time_spent_seconds``: Time spent on item.

.. Note::
    - This module is designed for general item-level interaction data.
    - Research analysis requires these variables. To use this analysis
      with your own data, ensure your dataset contains these columns
      (with compatible definitions).
    - The score collapsing function only modifies scores for items with
      more than three unique score points. For demonstration purposes
      only; simplifies analyses.

.. currentmodule:: utils.data_import

Functions
=========

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function_name_only.rst

    parse_process_data
    clean_parsed_data
    batch_process_zip
    collapse_scores_equal_freq

Standalone Execution
=====================
When run as a standalone script, this module processes interaction data,
cleans and merges response data, collapses scores, and outputs the results
to parquet format.

.. code-block:: bash

    python data_import.py

Output Files:
-------------
- ``COTS_2025_data.parquet``

.. Note::
   The following paths must be correctly set within the script's ``__main__``
   block for successful execution:

   - ``DATA_PATH``: Directory containing input and output data.
"""

__author__ = "William Muntean"
__email__ = "williamjmuntean@gmail.com"
__license__ = "GPL v3"
__maintainer__ = "William Muntean"
__date__ = "2025-08-22"


import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import yaml

ROOT_PATH = (
    Path("__file__").resolve().parents[0]
)  # 0 for .py or unsaved notebooks and 1 for .ipynb
sys.path.append(ROOT_PATH.as_posix())


def parse_process_data(xml_file):
    """
    Parses an interaction XML file and extracts item-level
    interaction data for a single exam.

    Parameters
    ----------
    xml_file : str or file-like
        Path to the XML file or file-like object containing interaction data.

    Returns
    -------
    dict
        Dictionary containing person ID and a list of item-level
        interaction data.

        - ``person_id``: Person identifier.
        - ``items``: List of dictionaries with item interaction metrics.
    """
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Define namespace
    with open(ROOT_PATH / "utils" / "config.yaml", "r") as f:
        config = yaml.safe_load(f)
    ns = {config["xml"]["prefix"]: config["xml"]["namespace"]}
    person_var = config["person"]["var"]

    # Extract person ID
    person_id = root.find(person_var).text

    # Initialize variables
    items_data = []
    current_item = None
    item_start_time = None
    interactions = []
    exhibit_interactions = []

    # Process all events
    for event in root.findall(".//event", ns):
        action = event.get("action")
        event_time = datetime.fromisoformat(event.get("time").replace("Z", "+00:00"))
        event_id = event.get("id")

        # Track item entry
        if action == "itemEntered":
            # Save previous item data if exists
            if current_item:
                time_spent = (event_time - item_start_time).total_seconds()
                items_data.append(
                    {
                        "item_id": current_item,
                        "interaction_count": len(interactions),
                        "exhibit_interaction_count": len(exhibit_interactions),
                        "has_exhibit": len(exhibit_interactions) > 0,
                        "response_changes": sum(
                            1
                            for i in interactions
                            if i.get("action") == "responseSelected"
                            and "-FEEDBACK" not in i.get("id", "")
                        ),
                        "time_spent_seconds": time_spent,
                        "interactions": interactions,
                        "exhibit_interactions": exhibit_interactions,
                    }
                )

            # Start tracking new item
            current_item = event_id
            item_start_time = event_time
            interactions = []
            exhibit_interactions = []

        # Track all interactions within the current item
        if current_item:
            interaction_data = {
                "action": action,
                "time": event_time,
                "id": event_id,
                "status": event.get("status"),
                "choice": event.get("choice"),
                "response_type": event.get("responseType"),
            }

            # Separate exhibit interactions from regular interactions
            if event_id:
                if "-feedback" in event_id.lower():
                    exhibit_interactions.append(interaction_data)
                else:
                    interactions.append(interaction_data)
            else:
                interactions.append(interaction_data)

    # Add the last item
    if current_item:
        items_data.append(
            {
                "item_id": current_item,
                "interaction_count": len(interactions),
                "exhibit_interaction_count": len(exhibit_interactions),
                "has_exhibit": len(exhibit_interactions) > 0,
                "response_changes": sum(
                    1
                    for i in interactions
                    if i.get("action") == "responseSelected"
                    and "-FEEDBACK" not in i.get("id", "")
                ),
                "time_spent_seconds": 0,  # Can't calculate for last item
                "interactions": interactions,
                "exhibit_interactions": exhibit_interactions,
            }
        )

    return {"person_id": person_id, "items": items_data}


def clean_parsed_data(parsed_data):
    """
    Cleans parsed interaction data and returns a modeling DataFrame.

    Parameters
    ----------
    parsed_data : dict
        Parsed interaction data from ``parse_process_data``.

    Returns
    -------
    pd.DataFrame
        DataFrame with item-level interaction metrics.
        - ``person_id``: Person identifier.
        - ``item_id``: Item identifier.
        - ``total_interactions``: Number of interactions per item.
        - ``exhibit_interactions``: Number of exhibit interactions.
        - ``has_exhibit``: Boolean, whether item has exhibit.
        - ``response_selections``: Number of response selections.
        - ``response_changes``: Number of response changes.
        - ``time_spent_seconds``: Time spent on item.
    """
    rows = []

    for item in parsed_data["items"]:
        # Count specific interaction types
        response_selections = sum(
            1 for i in item["interactions"] if i["action"] == "responseSelected"
        )

        # Count response changes (when a selection is turned off)
        response_changes = sum(
            1
            for i in item["interactions"]
            if i["action"] == "responseSelected" and i.get("status") == "off"
        )

        rows.append(
            {
                "person_id": parsed_data["person_id"],
                "item_id": item["item_id"],
                "total_interactions": item["interaction_count"],
                "exhibit_interactions": item["exhibit_interaction_count"],
                "has_exhibit": item["has_exhibit"],
                "response_selections": response_selections,
                "response_changes": response_changes,
                "time_spent_seconds": item["time_spent_seconds"],
            }
        )

    return pd.DataFrame(rows)


def batch_process_zip(process_path: Path) -> pd.DataFrame:
    """
    Unzips each file matching ``*.zip`` in ``process_path`` and processes
    the xml file inside.

    Parameters
    ----------
    uih_path : Path
        Path to the directory containing zip files.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of all processed interaction XMLs.

        - ``person_id``: Person identifier.
        - ``item_id``: Item identifier.
        - ``total_interactions``: Number of interactions per item.
        - ``exhibit_interactions``: Number of exhibit interactions.
        - ``has_exhibit``: Boolean, whether item has exhibit.
        - ``response_selections``: Number of response selections.
        - ``response_changes``: Number of response changes.
        - ``time_spent_seconds``: Time spent on item.
    """
    dfs = []
    for zip_file in process_path.glob("*.zip"):
        with ZipFile(zip_file) as zf:
            if "3.exam.interaction.xml" in zf.namelist():
                with zf.open("3.exam.interaction.xml") as xml_file:
                    parsed = parse_process_data(xml_file)
                    df = clean_parsed_data(parsed)
                    dfs.append(df)
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        return df
    else:
        return pd.DataFrame()


def collapse_scores_equal_freq(
    df: pd.DataFrame, item_col: str = "item_id", score_col: str = "score"
) -> pd.DataFrame:
    """
    Collapse score points for each item into no more than 3 categories
    with roughly equal observation counts per category.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing ``item_id`` and ``score`` columns.
    item_col : str, optional
        Name of the item column. Default is ``item_id``.
    score_col : str, optional
        Name of the score column. Default is ``score``.

    Returns
    -------
    pd.DataFrame
        DataFrame with collapsed score (0, 1, or 2) replacing score_col name.
    """
    df = df.copy()

    def bin_scores(s: pd.Series) -> pd.Series:
        if s.nunique() <= 3:
            # Already 3 or fewer unique scores, keep as is
            return s
        try:
            return pd.qcut(s, q=3, labels=False, duplicates="drop")
        except ValueError:
            return s

    df[score_col] = df.groupby(item_col)[score_col].transform(bin_scores)
    return df


if __name__ == "__main__":
    import sys

    ROOT_PATH = (
        Path("__file__").resolve().parents[0]
    )  # 0 for .py or unsaved notebooks and 1 for .ipynb
    sys.path.append(ROOT_PATH.as_posix())

    DATA_PATH = ROOT_PATH / "data"
    RESULTS_PATH = ROOT_PATH / "results"

    process_path = Path(DATA_PATH / "process_data")
    df_process = batch_process_zip(process_path)

    # Clean Response Data
    df_response = pd.read_csv(DATA_PATH / "Scored_2104_RN_ENU.csv.zip")
    df_response = df_response[
        [
            "RegistrationID",
            "Identifier_Item",
            "TimeSpent_Sec",
            "FinalTheta",
            "ItemSetID",
            "ItemType",
            "ScorePts",
        ]
    ]

    df_response = df_response.rename(
        columns={
            "RegistrationID": "person_id",
            "Identifier_Item": "item_id",
            "TimeSpent_Sec": "rt",
            "FinalTheta": "op_theta",
            "ItemSetID": "itemset_id",
            "ItemType": "item_type",
            "ScorePts": "score",
        }
    )
    df_response["person_id"] = df_response["person_id"].astype(str)
    df_response["item_id"] = df_response["item_id"].astype(str)

    df = df_response.merge(
        df_process,
        left_on=["person_id", "item_id"],
        right_on=["person_id", "item_id"],
        how="inner",
    )

    # Clean process data error
    mean_exhibit_interactions = df.groupby("item_id")["exhibit_interactions"].transform(
        "mean"
    )
    mask = mean_exhibit_interactions > 0.009
    df.loc[mask, "has_exhibit"] = True
    df.loc[~mask, "exhibit_interactions"] = 0
    df["score"] = df["score"].astype(int)
    df = collapse_scores_equal_freq(df)
    df = df[df["item_type"] != "Drop_Cloze"]

    df["person_id"] = pd.factorize(df["person_id"])[0] + 1
    df["item_id"] = pd.factorize(df["item_id"])[0] + 1
    df["itemset_id"] = pd.factorize(df["itemset_id"])[0] + 1
    df["itemset_id"] = df["itemset_id"].replace(-1, pd.NA)

    df.to_parquet(DATA_PATH / "COTS_2025_data.parquet")
