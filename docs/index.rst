.. trustcv documentation master file

TrustCV - Trustworthy Cross-Validation Toolkit
==============================================

.. image:: https://img.shields.io/pypi/v/trustcv.svg
   :target: https://pypi.org/project/trustcv/
   
.. image:: https://img.shields.io/badge/python-3.8%2B-blue
   :target: https://www.python.org/downloads/

A comprehensive toolkit for proper cross-validation in medical machine learning.

Features
--------

* **Medical-Specific CV Methods**: Patient-aware, temporal, and grouped cross-validation
* **Data Leakage Detection**: Automatic detection of common pitfalls
* **Clinical Metrics**: Sensitivity, specificity, PPV, NPV, NNT with confidence intervals
* **Regulatory Compliance**: FDA and CE-ready validation reports
* **Interactive Tutorials**: Learn through hands-on examples

Quick Start
-----------

Installation::

    pip install trustcv

Basic usage::

    from trustcv import TrustCVValidator

    validator = TrustCVValidator(
        method='patient_grouped_kfold',
        n_splits=5,
        check_leakage=True
    )

    results = validator.validate(
        model=your_model,
        X=features,
        y=labels,
        groups=patient_ids
    )

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorials/index
   best_practices

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/validators
   api/splitters
   api/metrics
   api/checkers
   api/datasets

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   temporal_validation
   nested_cv
   regulatory_compliance
   clinical_metrics

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog
   authors

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`