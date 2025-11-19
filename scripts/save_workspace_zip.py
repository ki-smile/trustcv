import os
import zipfile
from datetime import datetime

root = os.getcwd()
stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
backup_dir = os.path.join(root, 'backup')
os.makedirs(backup_dir, exist_ok=True)
dst = os.path.join(backup_dir, f'trustcv-backup-{stamp}.zip')

exclude_dirs = {'.git', 'backup', '__pycache__'}

with zipfile.ZipFile(dst, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for folder, dirs, files in os.walk(root):
        rel_folder = os.path.relpath(folder, root)
        if rel_folder == '.':
            rel_folder = ''
        # prune excluded dirs
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for f in files:
            # skip pyc and temporary files
            if f.endswith(('.pyc', '.pyo')):
                continue
            abs_path = os.path.join(folder, f)
            rel_path = os.path.relpath(abs_path, root)
            if rel_path.startswith('backup' + os.sep) or rel_path.startswith('.git' + os.sep):
                continue
            zf.write(abs_path, rel_path)

print(dst)
