import json
from pathlib import Path

nb_path = Path('notebooks/03_Temporal_Medical_amir.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))

needle = "raise KeyError('roc_auc-like key not found')"
repl = "    return np.nan, np.nan\n"

modified = False
for cell in nb.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    src_list = cell.get('source', [])
    new_src_list = []
    changed_cell = False
    for line in src_list:
        if needle in line:
            leading = line[:line.find('r')]
            new_src_list.append(leading + repl)
            changed_cell = True
            modified = True
        else:
            new_src_list.append(line)
    if changed_cell:
        cell['source'] = new_src_list

if not modified:
    raise SystemExit('No get_auc KeyError raise found to replace.')

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')

