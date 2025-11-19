import json
from pathlib import Path

nb_path = Path('notebooks/03_Temporal_Medical_amir.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))

helper_src = [
    "# Helpers + data fallbacks for temporal CV visualization\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import inspect\n",
    "\n",
    "# Ensure X, y, groups, timestamps exist\n",
    "if 'X' not in globals():\n",
    "    if 'panel' in globals():\n",
    "        if 'X_cols' not in globals():\n",
    "            drop = {'date','timestamp','time','ticker','patient_id','ret','abs_ret_tomorrow','y'}\n",
    "            X_cols = [c for c in panel.columns if c not in drop and pd.api.types.is_numeric_dtype(panel[c])]\n",
    "        X = panel[X_cols].to_numpy()\n",
    "        y = panel['y'].to_numpy() if 'y' in panel.columns else np.zeros(len(panel), dtype=int)\n",
    "        groups = panel['ticker'].to_numpy() if 'ticker' in panel.columns else np.arange(len(panel))\n",
    "        timestamps = panel['date'].to_numpy() if 'date' in panel.columns else np.arange(len(panel))\n",
    "    else:\n",
    "        n = 2000\n",
    "        X = np.random.randn(n, 10)\n",
    "        y = (np.random.rand(n) > 0.7).astype(int)\n",
    "        groups = np.arange(n) % 50\n",
    "        timestamps = pd.date_range('2020-01-01', periods=n, freq='D')\n",
    "\n",
    "def split_supports_arg(cv, name: str) -> bool:\n",
    "    return name in inspect.signature(cv.split).parameters\n",
    "\n",
    "def visualize_cv(cv, X, y=None, groups=None, timestamps=None, title=None, max_splits=5, ax=None):\n",
    "    want = inspect.signature(cv.split).parameters\n",
    "    kw = {}\n",
    "    if 'y' in want: kw['y'] = y\n",
    "    if 'groups' in want and groups is not None: kw['groups'] = groups\n",
    "    if 'timestamps' in want and timestamps is not None: kw['timestamps'] = timestamps\n",
    "    splits = []\n",
    "    for i, (tr, te) in enumerate(cv.split(X, **kw)):\n",
    "        if i >= max_splits: break\n",
    "        tr = np.asarray(tr); te = np.asarray(te)\n",
    "        if tr.size and te.size: splits.append((tr, te))\n",
    "    if not splits:\n",
    "        print(f'{cv.__class__.__name__}: no valid folds to visualize.')\n",
    "        return\n",
    "    import matplotlib.pyplot as plt\n",
    "    if ax is None:\n",
    "        fig, ax = plt.subplots(figsize=(10, 6))\n",
    "    for i, (tr, te) in enumerate(splits):\n",
    "        ax.scatter(tr, np.full(tr.shape, i), s=8, color='#870052', label='Train' if i == 0 else '')\n",
    "        ax.scatter(te, np.full(te.shape, i), s=8, color='#FF876F', label='Test' if i == 0 else '')\n",
    "    ax.set_xlabel('Sample Index')\n",
    "    ax.set_ylabel('CV Fold')\n",
    "    ax.set_yticks(range(len(splits)))\n",
    "    ax.set_yticklabels([f'Fold {i+1}' for i in range(len(splits))])\n",
    "    ax.legend(loc='upper right')\n",
    "    ax.set_title(title or cv.__class__.__name__)\n",
]

# Insert at top if not already present (avoid duplicates)
already = any(cell.get('cell_type')=='code' and 'def visualize_cv' in ''.join(cell.get('source',[])) for cell in nb.get('cells',[]))
if not already:
    nb['cells'].insert(0, {
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': helper_src,
    })

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')

