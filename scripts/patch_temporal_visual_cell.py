import json
from pathlib import Path

nb_path = Path('notebooks/03_Temporal_Medical_amir.ipynb')
data = json.loads(nb_path.read_text(encoding='utf-8'))

needle = 'plot_cv_splits(pgts'
modified = False

replacement_tail = (
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

for cell in data.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    src_list = cell.get('source', [])
    text = ''.join(src_list)
    if needle in text:
        # Replace the three lines that plot with pgts with our block
        lines = text.splitlines(True)
        new_lines = []
        i = 0
        while i < len(lines):
            if 'fig, ax = plot_cv_splits(pgts' in lines[i]:
                # Skip this line and the immediate following set_title and plt.show()
                i += 1
                # Skip potential set_title line
                while i < len(lines) and ('set_title' in lines[i] or lines[i].strip()=='' or 'plt.show()' in lines[i]):
                    i += 1
                # Insert replacement block
                new_lines.append(replacement_tail)
                continue
            new_lines.append(lines[i])
            i += 1
        # Also ensure earlier calls use patient_ids for groups if present
        fixed = ''.join(new_lines).replace('groups=groups', 'groups=patient_ids')
        cell['source'] = fixed.splitlines(True)
        modified = True

if not modified:
    raise SystemExit('Target plotting call not found; no changes made.')

nb_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')

