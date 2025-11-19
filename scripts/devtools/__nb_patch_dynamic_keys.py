# -*- coding: utf-8 -*-
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
nb_path = ROOT / 'notebooks' / '01_CV_Basics.ipynb'
nb = nbf.read(nb_path.open('r', encoding='utf-8'), as_version=4)
changed = False
for cell in nb.cells:
    if cell.get('cell_type') == 'code' and "mean_key = 'test_accuracy'" in ''.join(cell.get('source','')):
        src = ''.join(cell['source'])
        src = src.replace("mean_key = 'test_accuracy'","mean_key = 'accuracy' if 'accuracy' in result.mean_scores else ('test_accuracy' if 'test_accuracy' in result.mean_scores else 'score')")
        # ensure std and scores use the same key variable
        src = src.replace("fold_scores = result.scores[mean_key]","scores_key = 'accuracy' if 'accuracy' in result.scores else ('test_accuracy' if 'test_accuracy' in result.scores else 'score')\nfold_scores = result.scores[scores_key]")
        cell['source'] = src
        changed = True

if changed:
    nbf.write(nb, nb_path.open('w', encoding='utf-8'))
    print('Patched dynamic key selection in 01_CV_Basics.ipynb')
else:
    print('No change needed')
