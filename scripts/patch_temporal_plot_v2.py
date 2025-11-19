import json, re
from pathlib import Path

nb_path = Path('notebooks/03_Temporal_Medical_amir.ipynb')
data = json.loads(nb_path.read_text(encoding='utf-8'))

start_pat = re.compile(r"^\s*for i, \(train_idx, test_idx\) in enumerate\(trustcv_cv\.split\(X, timestamps=timestamps\)\):\s*$", re.M)
end_pat = re.compile(r"^\s*axes\[1\]\.set_title\(.*\)\s*$", re.M)

replacement_block = (
    "for i, (train_idx, test_idx) in enumerate(trustcv_cv.split(X, timestamps=timestamps)):\n"
    "    # Ensure positional indexing using numpy arrays\n"
    "    train_idx = np.asarray(train_idx)\n"
    "    test_idx = np.asarray(test_idx)\n"
    "\n"
    "    # Compute gap safely (handles empty folds)\n"
    "    gap_start = int(train_idx.max()) + 1 if train_idx.size else 0\n"
    "    gap_end = int(test_idx.min()) if test_idx.size else gap_start\n"
    "\n"
    "    axes[1].fill_between(train_idx, i - 0.4, i + 0.4, color=colors[0], label=\"Train\" if i == 0 else \"\")\n"
    "    if gap_end > gap_start:\n"
    "        axes[1].fill_between(np.arange(gap_start, gap_end), i - 0.4, i + 0.4, color='grey', alpha=0.2, label=\"Purge Gap\" if i == 0 else \"\")\n"
    "    axes[1].fill_between(test_idx, i - 0.4, i + 0.4, color=colors[1], label=\"Test\" if i == 0 else \"\")\n"
)

modified = False
for cell in data.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    m_start = start_pat.search(src)
    if not m_start:
        continue
    m_end = end_pat.search(src, m_start.end())
    if not m_end:
        continue
    before = src[:m_start.start()]
    after = src[m_end.start():]  # keep the set_title line and after
    new_src = before + replacement_block + after
    cell['source'] = new_src.splitlines(True)
    modified = True

if not modified:
    raise SystemExit('Start/end markers not found; no changes made.')

nb_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')

