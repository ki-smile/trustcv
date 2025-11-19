# -*- coding: utf-8 -*-
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
nb_path = ROOT / 'notebooks' / '01_CV_Basics.ipynb'
nb = nbf.read(nb_path.open('r', encoding='utf-8'), as_version=4)

changed = False
for cell in nb.cells:
    if cell.get('cell_type') == 'code' and 'Trustworthy Cross-Validation Report' in ''.join(cell.get('source','')):
        src = ''.join(cell['source'])
        lines = src.splitlines()
        new_lines = []
        inserted_logic = False
        for ln in lines:
            if 'Performance (mean' in ln and "result.mean_scores['accuracy']" in ln:
                if not inserted_logic:
                    new_lines.append("acc_key_mean = 'accuracy' if 'accuracy' in result.mean_scores else ('test_accuracy' if 'test_accuracy' in result.mean_scores else 'score')")
                    new_lines.append("acc_key_std = 'accuracy' if 'accuracy' in result.std_scores else ('test_accuracy' if 'test_accuracy' in result.std_scores else 'score')")
                    new_lines.append("mean_acc = result.mean_scores[acc_key_mean]")
                    new_lines.append("std_acc = result.std_scores[acc_key_std]")
                    inserted_logic = True
                new_lines.append("print('Performance (mean +/- std): ' + f'{mean_acc:.3f} +/- {std_acc:.3f}')")
            elif "result.confidence_intervals['accuracy']" in ln:
                new_lines.append("ci_key = 'accuracy' if 'accuracy' in result.confidence_intervals else ('score' if 'score' in result.confidence_intervals else acc_key_mean.replace('test_',''))")
                new_lines.append("ci = result.confidence_intervals[ci_key]")
                new_lines.append("print('95% CI: ' + f'[{ci[0]:.3f}, {ci[1]:.3f}]')")
            elif "result.scores['accuracy']" in ln:
                new_lines.append("scores_key = 'accuracy' if 'accuracy' in result.scores else ('test_accuracy' if 'test_accuracy' in result.scores else 'score')")
                new_lines.append("print('\nFold Scores: ' + str([f'{s:.3f}' for s in result.scores[scores_key]]))")
            else:
                new_lines.append(ln)
        new_src = '\n'.join(new_lines)
        if new_src != src:
            cell['source'] = new_src
            changed = True

if changed:
    nbf.write(nb, nb_path.open('w', encoding='utf-8'))
    print('Patched 01_CV_Basics.ipynb to handle accuracy/test_accuracy keys.')
else:
    print('No changes needed in 01_CV_Basics.ipynb')
