# 0. Resolve this script's directory so it can be run from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Get the recipes (a shallow fetch is enough). Cloned to /tmp, not under this
#    directory, so the checkout never ends up inside the aiter tree. Pinned to
#    the commit InferenceX_<hash>/recipe_models.py was last verified against;
#    bump both together (new hash dir + this var) when picking up new recipes.
INFERENCEX_COMMIT=bde2579a49696924c7783519224fbe49e2c678b7
INFERENCEX_HASH_DIR="$SCRIPT_DIR/InferenceX_bde2579a"
rm -rf /tmp/InferenceX
mkdir -p /tmp/InferenceX
git -C /tmp/InferenceX init -q
git -C /tmp/InferenceX remote add origin https://github.com/SemiAnalysisAI/InferenceX
git -C /tmp/InferenceX fetch --depth 1 origin "$INFERENCEX_COMMIT"
git -C /tmp/InferenceX checkout -q FETCH_HEAD

# 2. Needs Python 3 + PyYAML
pip install pyyaml

# 3. Parse the AMD agentic recipes into one row per benchmark point. Generic
#    across InferenceX commits (schema-level parsing only, no model knowledge).
python3 "$SCRIPT_DIR/parse_recipes.py" \
  /tmp/InferenceX/inferencex-e2e/benchmarks/single_node/srt-slurm-recipes \
  /tmp/recipe_rows.json

# 4. Build the AITER-style JSON (written to ../../../utils/model_configs/model_configs_semi_agentx_<InferenceX commit>.json)
#    and the per-point CSV. Reads the commit hash from /tmp/InferenceX (override with INFERENCEX_DIR).
#    Model-specific knowledge (ARCH/RECIPE_MODEL/etc.) comes from the pinned
#    InferenceX_<hash>/recipe_models.py, not from this script.
python3 "$SCRIPT_DIR/build_model_configs.py" \
  /tmp/recipe_rows.json /tmp/agentx_points.csv \
  "$INFERENCEX_HASH_DIR/recipe_models.py"
