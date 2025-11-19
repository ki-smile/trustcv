import warnings

from trustcv.validators import MedicalValidator, TrustCVValidator


def test_medicalvalidator_deprecated_alias():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DeprecationWarning)
        m = MedicalValidator()
        assert any(isinstance(ww.message, DeprecationWarning) for ww in w)
        assert isinstance(m, TrustCVValidator)

