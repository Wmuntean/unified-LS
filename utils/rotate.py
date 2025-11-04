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
Latent Space Alignment Utilities
=======================================================

This module provides utility functions for aligning latent space coordinates
across multiple MCMC chains in latent space IRT models. The main functionality
is based on Procrustes rotation, enabling consistent interpretation of latent
dimensions across chains and facilitating downstream analysis and visualization.

Latent Space Alignment Process
==============================

The alignment process is performed in the following stages:

1. **Extraction**:

   - Extract latent coordinates for persons and items from Stan draws by chain.

2. **Alignment**:

   - Align latent spaces across chains using Procrustes analysis, referencing a
     selected chain for consistent orientation.

3. **Replacement**:

   - Replace original latent coordinates in Stan draws with aligned coordinates,
     preserving all other parameters and metadata.

.. Note::
    - All alignment is performed in-place on copies of the original draws.
    - Functions assume Stan output format with ``chain__`` column and parameter
      names in ``prefix[i,d]`` format.

.. Important::
    - Reference chain selection impacts orientation; use the default (chain 1)
      unless specific justification exists.

.. currentmodule:: utils.rotate

Functions
=========

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function_name_only.rst

    extract_latent_coordinates
    align_latent_spaces
    create_aligned_draws_dataframe
"""

__author__ = "William Muntean"
__email__ = "williamjmuntean@gmail.com"
__license__ = "GPL v3"
__maintainer__ = "William Muntean"
__date__ = "2025-08-29"

import numpy as np
import pandas as pd
from scipy.spatial import procrustes


def extract_latent_coordinates(
    df_draws: pd.DataFrame, n_entities: int, D: int, prefix: str
) -> dict[int, np.ndarray]:
    """
    Extract and reshape latent coordinates from Stan draws by chain.

    Parameters
    ----------
    df_draws : pd.DataFrame
        DataFrame containing Stan draws with ``chain__`` column.
    n_entities : int
        Number of entities (persons or items).
    D : int
        Number of latent dimensions.
    prefix : str
        Parameter prefix (e.g., 'xi' or 'zt_centered').

    Returns
    -------
    dict[int, np.ndarray]
        Dictionary mapping chain IDs to (n_entities, D) coordinate matrices.
    """
    chain_means = {}

    for chain_id in df_draws["chain__"].unique():
        chain_df = df_draws[df_draws["chain__"] == chain_id]
        mean_params = chain_df.drop(columns="chain__").mean()

        # Reshape into matrix
        matrix = np.zeros((n_entities, D))
        for i in range(n_entities):
            for d in range(D):
                param_name = f"{prefix}[{i + 1},{d + 1}]"
                matrix[i, d] = mean_params[param_name]

        chain_means[chain_id] = matrix

    return chain_means


def align_latent_spaces(
    person_coords: dict[int, np.ndarray],
    item_coords: dict[int, np.ndarray],
    ref_chain_id: int = 1,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Align latent spaces across chains using Procrustes analysis.

    Parameters
    ----------
    person_coords : dict[int, np.ndarray]
        Person coordinates by chain.
    item_coords : dict[int, np.ndarray]
        Item coordinates by chain.
    ref_chain_id : int, optional
        Reference chain ID for alignment. Default is 1.

    Returns
    -------
    tuple[dict[int, np.ndarray], dict[int, np.ndarray]]
        Aligned person and item coordinates by chain.
    """
    # Create reference matrix
    reference_matrix = np.vstack(
        [person_coords[ref_chain_id], item_coords[ref_chain_id]]
    )

    aligned_person_coords = {ref_chain_id: person_coords[ref_chain_id]}
    aligned_item_coords = {ref_chain_id: item_coords[ref_chain_id]}

    n_persons = person_coords[ref_chain_id].shape[0]

    for chain_id in person_coords.keys():
        if chain_id == ref_chain_id:
            continue

        # Combine current chain coordinates
        current_matrix = np.vstack([person_coords[chain_id], item_coords[chain_id]])

        # Perform Procrustes alignment
        _, aligned_matrix, disparity = procrustes(reference_matrix, current_matrix)

        print(
            f"Aligning Chain {chain_id} to Chain {ref_chain_id}. "
            f"Disparity: {disparity:.4f}"
        )

        # Split back into person and item parts
        aligned_person_coords[chain_id] = aligned_matrix[:n_persons, :]
        aligned_item_coords[chain_id] = aligned_matrix[n_persons:, :]

    return aligned_person_coords, aligned_item_coords


def create_aligned_draws_dataframe(
    original_draws: pd.DataFrame,
    aligned_person_coords: dict[int, np.ndarray],
    aligned_item_coords: dict[int, np.ndarray],
    D: int,
) -> pd.DataFrame:
    """
    Create DataFrame with aligned latent coordinates replacing original draws.

    Parameters
    ----------
    original_draws : pd.DataFrame
        Original Stan draws DataFrame.
    aligned_person_coords : dict[int, np.ndarray]
        Aligned person coordinates by chain.
    aligned_item_coords : dict[int, np.ndarray]
        Aligned item coordinates by chain.
    D : int
        Number of latent dimensions.

    Returns
    -------
    pd.DataFrame
        DataFrame with aligned coordinates replacing original latent parameters:

        - Replaces all ``xi[i,d]`` parameters with aligned person coordinates.
        - Replaces all ``zt_centered[i,d]`` parameters with aligned item coordinates.
        - Preserves all other parameters and metadata columns.
    """
    aligned_draws = original_draws.copy()

    # Replace person coordinates
    for chain_id, coords in aligned_person_coords.items():
        chain_mask = aligned_draws["chain__"] == chain_id
        n_persons = coords.shape[0]

        for i in range(n_persons):
            for d in range(D):
                param_name = f"xi[{i + 1},{d + 1}]"
                if param_name in aligned_draws.columns:
                    aligned_draws.loc[chain_mask, param_name] = coords[i, d]

    # Replace item coordinates
    for chain_id, coords in aligned_item_coords.items():
        chain_mask = aligned_draws["chain__"] == chain_id
        n_items = coords.shape[0]

        for i in range(n_items):
            for d in range(D):
                param_name = f"zt_centered[{i + 1},{d + 1}]"
                if param_name in aligned_draws.columns:
                    aligned_draws.loc[chain_mask, param_name] = coords[i, d]

    return aligned_draws
