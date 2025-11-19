import json
from pathlib import Path

nb_path = Path('notebooks/03_Temporal_Medical_amir.ipynb')
data = json.loads(nb_path.read_text(encoding='utf-8'))

target_snippet = 'for i, (train_idx, test_idx) in enumerate(trustcv_cv.split(X, timestamps=timestamps)):'
modified = False

for cell in data.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    src = cell.get('source', [])
    text = ''.join(src)
    if target_snippet in text:
        # Replace the body of the loop with a robust version
        lines = text.splitlines(True)
        new_lines = []
        in_loop = False
        indent = ''
        for ln in lines:
            new_lines.append(ln)
            if target_snippet in ln:
                # Determine indentation
                indent = ln.split('for')[0]
                in_loop = True
                # After the for-line, inject our robust block and skip following original lines until next 'axes[1].set_title' or blank line
                new_block = (
                    f"{indent}    # Ensure positional indexing using numpy arrays\n"
                    f"{indent}    train_idx = np.asarray(train_idx)\n"
                    f"{indent}    test_idx = np.asarray(test_idx)\n"
                    f"{indent}\n"
                    f"{indent}    # Compute gap safely (handles empty folds)\n"
                    f"{indent}    gap_start = int(train_idx.max()) + 1 if train_idx.size else 0\n"
                    f"{indent}    gap_end = int(test_idx.min()) if test_idx.size else gap_start\n"
                    f"{indent}\n"
                    f"{indent}    axes[1].fill_between(train_idx, i - 0.4, i + 0.4, color=colors[0], label=\"Train\" if i == 0 else \"\")\n"
                    f"{indent}    if gap_end > gap_start:\n"
                    f"{indent}        axes[1].fill_between(np.arange(gap_start, gap_end), i - 0.4, i + 0.4, color='grey', alpha=0.2, label=\"Purge Gap\" if i == 0 else \"\")\n"
                    f"{indent}    axes[1].fill_between(test_idx, i - 0.4, i + 0.4, color=colors[1], label=\"Test\" if i == 0 else \"\")\n"
                )
                new_lines.append(new_block)
                continue
            if in_loop:
                # Skip old body lines until we reach a set_title or dedented block
                # We detect lines that include 'axes[1].set_title' and re-include them; otherwise, we skip
                if 'axes[1].set_title' in ln:
                    new_lines.append(ln)
                    in_loop = False
                else:
                    # Do not append old content
                    pass
        cell['source'] = new_lines
        modified = True

if not modified:
    raise SystemExit('Did not find the plotting loop to patch.')

nb_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')

