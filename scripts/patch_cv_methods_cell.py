import json
from pathlib import Path

nb_path = Path('notebooks/03_Temporal_Medical_amir.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))

target_snippets = (
    'BlockedTimeSeriesSplit',
    'ExpandingWindowCV(n_splits',
    'RollingWindowCV(n_splits',
)

new_source = [
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "from trustcv.splitters.temporal import (\n",
    "    ExpandingWindowCV,\n",
    "    BlockedTimeSeries,\n",
    "    RollingWindowCV,\n",
    "    PurgedKFoldCV,\n",
    "    CombinatorialPurgedCV,\n",
    "    PurgedGroupTimeSeriesSplit,\n",
    ")\n",
    "\n",
    "plt.style.use('seaborn-v0_8-whitegrid')\n",
    "colors = ['#870052', '#FF876F', '#4F0433', '#2196F3', '#EDF4F4']\n",
    "sns.set_palette(colors)\n",
    "\n",
    "# Initialize CV methods with correct constructor arguments\n",
    "cv_methods = {\n",
    "    \"ExpandingWindowCV\": ExpandingWindowCV(\n",
    "        min_train_size=4000,\n",
    "        forecast_horizon=1500,\n",
    "        step_size=100,\n",
    "        gap=100\n",
    "    ),\n",
    "    \"BlockedTimeSeries\": BlockedTimeSeries(\n",
    "        n_splits=5,\n",
    "        block_size=150\n",
    "    ),\n",
    "    \"RollingWindowCV\": RollingWindowCV(\n",
    "        window_size=4000,\n",
    "        forecast_horizon=1500,\n",
    "        step_size=100,\n",
    "        gap=100\n",
    "    ),\n",
    "    \"PurgedKFoldCV\": PurgedKFoldCV(\n",
    "        n_splits=5,\n",
    "        purge_gap=100,\n",
    "        embargo_pct=0.01\n",
    "    ),\n",
    "    \"CombinatorialPurgedCV\": CombinatorialPurgedCV(\n",
    "        n_splits=5,\n",
    "        n_test_groups=2,\n",
    "        purge_gap=100,\n",
    "        embargo_pct=0.01\n",
    "    ),\n",
    "    \"PurgedGroupTimeSeriesSplit\": PurgedGroupTimeSeriesSplit(\n",
    "        n_splits=5,\n",
    "        purge_gap=200,\n",
    "        embargo_size=0.01\n",
    "    ),\n",
    "}\n",
]

modified = False
for cell in nb.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if any(sn in src for sn in target_snippets) and 'cv_methods' in src:
        cell['source'] = new_source
        modified = True
        break

if not modified:
    raise SystemExit('Did not find a cv_methods cell with incorrect constructors to replace.')

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')

