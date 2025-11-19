import json
from pathlib import Path

nb_path = Path('notebooks/03_Temporal_Medical_amir.ipynb')
data = json.loads(nb_path.read_text(encoding='utf-8'))

replacement_block = (
    "# Custom visualization for PurgedGroupTimeSeriesSplit (needs timestamps)\n"
    "fig, ax = plt.subplots(figsize=(10,6))\n"
    "splits = list(pgts.split(X, y=y, groups=patient_ids, timestamps=timestamps))\n"
    "for i, (train_idx, test_idx) in enumerate(splits):\n"
    "    ax.scatter(train_idx, [i]*len(train_idx), color='tab:blue', s=8, label='Train' if i==0 else '')\n"
    "    ax.scatter(test_idx, [i]*len(test_idx), color='tab:red', s=8, label='Test' if i==0 else '')\n"
    "ax.set_xlabel('Sample Index')\n"
    "ax.set_ylabel('CV Fold')\n"
    "ax.set_yticks(range(len(splits)))\n"
    "ax.set_yticklabels([f'Fold {i+1}' for i in range(len(splits))])\n"
    "ax.legend(loc='upper right')\n"
    "ax.set_title('PurgedGroupTimeSeriesSplit (time + groups + purge + embargo)')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
)

modified = False
for cell in data.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    src = cell.get('source', [])
    new_src = []
    i = 0
    while i < len(src):
        line = src[i]
        if 'fig, ax = plot_cv_splits(pgts' in line:
            # Skip this and next lines until we pass title/show lines
            i += 1
            while i < len(src) and (('set_title' in src[i]) or ('plt.show()' in src[i]) or (src[i].strip()=='')):
                i += 1
            new_src.append(replacement_block)
            modified = True
            continue
        # Also fix groups variable name in this cell
        new_src.append(line.replace('groups=groups', 'groups=patient_ids'))
        i += 1
    cell['source'] = new_src

if not modified:
    raise SystemExit('No pgts plotting call found to replace.')

nb_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')

