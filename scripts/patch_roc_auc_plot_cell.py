import json
from pathlib import Path

nb_path = Path('notebooks/03_Temporal_Medical_amir.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))

needle = "'roc_auc_mean': [res_kf['roc_auc'][0]"
replaced = False

new_cell_src = [
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import re\n",
    "\n",
    "# --- Visualization settings ---\n",
    "plt.style.use('seaborn-v0_8-whitegrid')\n",
    "colors = ['#870052', '#FF876F', '#4F0433', '#EDF4F4']\n",
    "sns.set_palette(colors)\n",
    "\n",
    "def extract_mean_std(val):\n",
    "    if isinstance(val, str):\n",
    "        nums = [float(x) for x in re.findall(r'[0-9]*\\.?[0-9]+', val)]\n",
    "        if len(nums) >= 2: return nums[0], nums[1]\n",
    "        if len(nums) == 1: return nums[0], 0.0\n",
    "        return np.nan, np.nan\n",
    "    try:\n",
    "        arr = np.asarray(val, dtype=float)\n",
    "        if arr.ndim == 1:\n",
    "            if arr.size == 2 and np.isscalar(arr[0]) and np.isscalar(arr[1]):\n",
    "                return float(arr[0]), float(arr[1])\n",
    "            return float(arr.mean()), float(arr.std())\n",
    "    except Exception:\n",
    "        pass\n",
    "    if np.isscalar(val):\n",
    "        return float(val), 0.0\n",
    "    return np.nan, np.nan\n",
    "\n",
    "def get_auc(res):\n",
    "    # Try common keys\n",
    "    for k in ('roc_auc','auc','roc_auc_score'):\n",
    "        if isinstance(res, dict) and k in res:\n",
    "            return extract_mean_std(res[k])\n",
    "    # Search nested dicts\n",
    "    if isinstance(res, dict):\n",
    "        for v in res.values():\n",
    "            if isinstance(v, dict):\n",
    "                try:\n",
    "                    return get_auc(v)\n",
    "                except Exception:\n",
    "                    continue\n",
    "    # If not found, return NaNs so plotting can proceed gracefully\n",
    "    return np.nan, np.nan\n",
    "\n",
    "m_kf, s_kf = get_auc(res_kf)\n",
    "m_ts, s_ts = get_auc(res_ts)\n",
    "m_pg, s_pg = get_auc(res_pg)\n",
    "\n",
    "results_df = pd.DataFrame({\n",
    "    'Method': [\n",
    "        'Naïve KFold\\n(Leaks Time & Group Data)',\n",
    "        'TimeSeriesSplit\\n(Respects Time)',\n",
    "        'PurgedGroupTimeSeriesSplit\\n(Respects Time & Groups)'\n",
    "    ],\n",
    "    'roc_auc_mean': [m_kf, m_ts, m_pg],\n",
    "    'roc_auc_std':  [s_kf, s_ts, s_pg],\n",
    "})\n",
    "\n",
    "# --- Plotting ---\n",
    "plt.figure(figsize=(12, 7))\n",
    "bars = plt.bar(\n",
    "    results_df['Method'],\n",
    "    results_df['roc_auc_mean'],\n",
    "    yerr=results_df['roc_auc_std'],\n",
    "    capsize=5,\n",
    "    color=[colors[1], colors[0], colors[2]]\n",
    ")\n",
    "\n",
    "plt.title('Comparison of Cross-Validation Methods on Financial Panel Data', fontsize=16, fontweight='bold')\n",
    "plt.ylabel('ROC AUC Score', fontsize=12)\n",
    "try:\n",
    "    ymax = float(np.nanmax(results_df['roc_auc_mean']))\n",
    "    plt.ylim(0.5, ymax * 1.1)\n",
    "except Exception:\n",
    "    pass\n",
    "plt.xticks(rotation=0, fontsize=11)\n",
    "\n",
    "# Add value labels\n",
    "for bar in bars:\n",
    "    yval = bar.get_height()\n",
    "    if pd.notna(yval):\n",
    "        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')\n",
    "\n",
    "# Annotate first bar if numeric\n",
    "if pd.notna(results_df['roc_auc_mean'].iloc[0]):\n",
    "    plt.annotate(\n",
    "        'Artificially high score due to data leakage!\\n(Model is \"cheating\" by seeing future data)',\n",
    "        xy=(0, results_df['roc_auc_mean'].iloc[0]),\n",
    "        xytext=(0.5, results_df['roc_auc_mean'].iloc[0] + 0.05),\n",
    "        arrowprops=dict(facecolor='red', shrink=0.05, alpha=0.7),\n",
    "        ha='center', fontsize=12, color='red'\n",
    "    )\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
]

for cell in nb.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if needle in src:
        cell['source'] = new_cell_src
        replaced = True
        break

if not replaced:
    raise SystemExit('Target plotting cell not found; no changes made.')

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
