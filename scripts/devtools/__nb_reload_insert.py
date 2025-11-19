import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
nb_path = ROOT / 'notebooks' / '01_CV_Basics.ipynb'
nb = nbf.read(nb_path.open('r', encoding='utf-8'), as_version=4)

# Insert a reload cell at the top if not present
reload_code = (
    "# Ensure latest trustcv code is loaded\n"
    "from importlib import reload\n"
    "import trustcv.validators as _tcv_validators\n"
    "reload(_tcv_validators)\n"
    "from trustcv import MedicalValidator\n"
)

needs_insert = True
for cell in nb.cells[:3]:
    if cell.get('cell_type') == 'code' and 'reload(_tcv_validators)' in ''.join(cell.get('source','')):
        needs_insert = False
        break

if needs_insert:
    nb.cells.insert(0, nbf.v4.new_code_cell(reload_code))
    nbf.write(nb, nb_path.open('w', encoding='utf-8'))
    print('Inserted reload cell into 01_CV_Basics.ipynb')
else:
    print('Reload cell already present')
