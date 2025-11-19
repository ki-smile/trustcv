import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
notebooks_dir = ROOT / 'notebooks'
fixed = []
for nb_path in sorted(notebooks_dir.glob('*.ipynb')):
    nb = nbf.read(nb_path.open('r', encoding='utf-8'), as_version=4)
    changed = False
    for cell in nb.cells:
        if cell.get('cell_type') == 'code':
            if cell.get('outputs'):
                cell['outputs'] = []
                changed = True
            if cell.get('execution_count') is not None:
                cell['execution_count'] = None
                changed = True
            src = cell.get('source','')
            new_src = src
            new_src = new_src.replace('PatientGroupKFold', 'GroupKFoldMedical')
            new_src = new_src.replace('TemporalClinical', 'TimeSeriesSplit')
            new_src = new_src.replace('from trustcv import validate_medical_model',
                                      'from trustcv.reporting.regulatory_report import validate_medical_model')
            if 'MedicalValidatorDemo' in new_src and 'class MedicalValidatorDemo' not in new_src:
                new_src = new_src.replace('MedicalValidator)class MedicalValidatorDemo',
                                          'MedicalValidator)\n\nclass MedicalValidatorDemo')
            if new_src != src:
                cell['source'] = new_src
                changed = True
    if changed:
        nbf.write(nb, nb_path.open('w', encoding='utf-8'))
        fixed.append(nb_path.name)
print('Cleared and patched:', ', '.join(fixed) if fixed else 'none')
