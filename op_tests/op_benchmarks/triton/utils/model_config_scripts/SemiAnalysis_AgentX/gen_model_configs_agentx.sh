# 1. Get the recipes (a shallow clone is enough). Cloned to /tmp, not under this
#    directory, so the checkout never ends up inside the aiter tree.
rm -rf /tmp/InferenceX
git clone --depth 1 https://github.com/SemiAnalysisAI/InferenceX /tmp/InferenceX

# 2. Needs Python 3 + PyYAML
pip install pyyaml

# 3. Parse the AMD agentic recipes into one row per benchmark point
python3 parse_recipes.py \
  /tmp/InferenceX/inferencex-e2e/benchmarks/single_node/srt-slurm-recipes \
  /tmp/recipe_rows.json

# 4. Build the AITER-style JSON (written to ../../../utils/model_configs/model_configs_semi_agentx_<InferenceX commit>.json)
#    and the per-point CSV. Reads the commit hash from /tmp/InferenceX (override with INFERENCEX_DIR).
python3 build_model_configs.py /tmp/recipe_rows.json /tmp/agentx_points.csv
