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
Run LS-PCM models and save results.
"""

__author__ = "William Muntean"
__email__ = "williamjmuntean@gmail.com"
__license__ = "GPL v3"
__maintainer__ = "William Muntean"
__date__ = "2025-08-29"

import sys
import time
from pathlib import Path

import cmdstanpy

ROOT_PATH = (
    Path("__file__").resolve().parents[0]
)  # 0 for .py or unsaved notebooks and 1 for .ipynb
sys.path.append(ROOT_PATH.as_posix())
from utils import rotate

DATA_PATH = ROOT_PATH / "data"
RESULTS_PATH = ROOT_PATH / "results"
MODEL_PATH = ROOT_PATH / "analysis" / "models"


def run_stan_model(model_path: Path | str, run_name: Path | str, stan_data: dict):
    model = cmdstanpy.CmdStanModel(
        stan_file=model_path, cpp_options={"STAN_THREADS": True}
    )

    init_fit = model.pathfinder(data=stan_data, num_paths=4, draws=1000)

    inits = init_fit.create_inits()

    fit = model.sample(
        data=stan_data,
        chains=4,
        parallel_chains=4,
        iter_warmup=100,
        iter_sampling=800,
        show_progress=True,
        inits=inits,
    )

    run_path = RESULTS_PATH / run_name
    if run_path.exists():
        timestamp = time.strftime("%Y%m%d-%H%M")
        run_path = RESULTS_PATH / f"ls-pcm_{timestamp}"

    fit.save_csvfiles(dir=run_path)

    df_fit = fit.draws_pd()
    df_fit.to_parquet(run_path / "{run_path.name}.parquet")

    # Align latent space ------ START --------------------------------
    person_vars = [var for var in df_fit.columns if "xi[" in var]
    item_vars = [var for var in df_fit.columns if "zt_centered[" in var]

    person_draws = df_fit[["chain__"] + person_vars]
    item_draws = df_fit[["chain__"] + item_vars]

    person_chain_coords = rotate.extract_latent_coordinates(
        person_draws, stan_data["n_persons"], stan_data["D"], "xi"
    )
    item_chain_coords = rotate.extract_latent_coordinates(
        item_draws, stan_data["n_items"], stan_data["D"], "zt_centered"
    )
    aligned_person_coords, aligned_item_coords = rotate.align_latent_spaces(
        person_chain_coords, item_chain_coords
    )
    df_fit_aligned = rotate.create_aligned_draws_dataframe(
        df_fit, aligned_person_coords, aligned_item_coords, D=2
    )

    df_fit_aligned.to_parquet(run_path / "{run_path.name}_aligned.parquet")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    import cmdstanpy
    import numpy as np
    import pandas as pd

    ROOT_PATH = (
        Path("__file__").resolve().parents[0]
    )  # 0 for .py or unsaved notebooks and 1 for .ipynb
    sys.path.append(ROOT_PATH.as_posix())

    DATA_PATH = ROOT_PATH / "data"
    RESULTS_PATH = ROOT_PATH / "results"
    MODEL_PATH = ROOT_PATH / "analysis" / "models"

    df_resp = pd.read_parquet(DATA_PATH / "COTS_2025_data.parquet")
    df_resp["max_score"] = df_resp.groupby("item_id")["score"].transform("max")
    df_resp["log_rt"] = np.log(df_resp["rt"])

    # Ensure 1-index for Stan
    df_resp["person_id"] = pd.factorize(df_resp["person_id"])[0] + 1
    df_resp["item_id"] = pd.factorize(df_resp["item_id"])[0] + 1

    # Run LS-PCM ------ START --------------------------------
    threshold_start = df_resp.groupby("item_id")["max_score"].first().cumsum() + 1
    threshold_start = threshold_start.shift(1).replace(np.nan, 1).astype(int).to_numpy()
    total_thresholds = df_resp.groupby("item_id")["max_score"].first().sum()

    stan_data = {
        "N": len(df_resp),
        "item_id": df_resp["item_id"].to_numpy(),
        "person_id": df_resp["person_id"].to_numpy(),
        "theta": df_resp["op_theta"].to_numpy(),
        "scores": (df_resp["score"] + 1).to_numpy(),
        "categories_per_item": (df_resp["max_score"] + 1).to_numpy(),
        "n_items": df_resp["item_id"].nunique(),
        "threshold_start": threshold_start,
        "total_thresholds": total_thresholds,
        "max_categories": df_resp["max_score"].max() + 1,
    }

    stan_model = MODEL_PATH / "ls_pcm_fixed_theta.stan"
    # run_stan_model(model_path=stan_model, run_name="ls-pcm", stan_data=stan_data)

    # Run LS-LNRT ------ START --------------------------------
    stan_data = {
        "N": len(df_resp),
        "n_items": df_resp["item_id"].nunique(),
        "n_persons": df_resp["person_id"].nunique(),
        "D": 2,
        "item_id": df_resp["item_id"].to_numpy(),
        "person_id": df_resp["person_id"].to_numpy(),
        "log_rt": df_resp["log_rt"].to_numpy(),
    }

    stan_model = MODEL_PATH / "ls_LNRT.stan"
    run_stan_model(model_path=stan_model, run_name="ls-lnrt", stan_data=stan_data)
