import json
from pathlib import Path

nb_path = Path('notebooks/02_Patient_Level_CV.ipynb')
data = json.loads(nb_path.read_text(encoding='utf-8'))

modified = False
for cell in data.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    src = cell.get('source', [])
    text = ''.join(src)
    if 'import xgboost as xgb' in text:
        new_src = [
            "from sklearn.ensemble import RandomForestClassifier\n",
            "from sklearn.model_selection import KFold, cross_val_score\n",
            "\n",
            "# Optional: XGBoost\n",
            "try:\n",
            "    import xgboost as xgb\n",
            "    XGB_AVAILABLE = True\n",
            "except ModuleNotFoundError:\n",
            "    XGB_AVAILABLE = False\n",
            "    print(\"xgboost not installed; skipping XGBoost examples.\")\n",
            "\n",
            "# Optional: PyTorch\n",
            "try:\n",
            "    import torch\n",
            "    import torch.nn as nn\n",
            "    from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler\n",
            "    TORCH_AVAILABLE = True\n",
            "except ModuleNotFoundError:\n",
            "    TORCH_AVAILABLE = False\n",
            "    print(\"PyTorch not installed; skipping Torch examples.\")\n",
        ]
        cell['source'] = new_src
        modified = True

if modified:
    nb_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
else:
    raise SystemExit('No cell with xgboost import found; no changes made.')

