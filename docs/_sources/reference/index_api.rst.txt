API Reference
=============

Documentation for all the main modules and functions of the unified_ls package.

This API reference documents the main scripts and modules used in the unified latent space research project.  
The repository supports a unified Item Response Theory (IRT) framework for educational and psychological measurement, integrating response accuracy, response time, and interaction/process data.  
This project provides a collection of research scripts, modeling utilities, and analysis notebooks designed for reproducible scientific workflows.

**Sections:**

- **Data Import:** Tools for parsing, cleaning, and structuring raw item-level data from XML and zipped sources into unified DataFrames for modeling and analysis.
- **Utilities:** Functions for latent space alignment (e.g., Procrustes analysis), coordinate extraction, and other shared helpers for post-processing model outputs.
- **Analysis:** Scripts for fitting Stan latent space models (IRT, RT, process, unified), saving results, and performing downstream analyses.

Each section below includes documentation and usage notes, with links to relevant notebooks and scripts for practical examples.


.. rubric:: Data Import

.. include:: ../../../README.md
   :parser: myst_parser.sphinx_
   :start-after: <!-- data_import -->
   :end-before: <!-- rotate -->

.. rubric:: Utilities

.. include:: ../../../README.md
   :parser: myst_parser.sphinx_
   :start-after: <!-- rotate -->
   :end-before: <!-- run_ls_models -->

.. rubric:: Analysis

.. include:: ../../../README.md
   :parser: myst_parser.sphinx_
   :start-after: <!-- run_ls_models -->
   :end-before: <!-- end modules -->

.. include:: ../../../README.md
   :parser: myst_parser.sphinx_
   :start-after: <!-- figures -->
   :end-before: <!-- end analysis -->

.. toctree::
   :maxdepth: 2
   :hidden:
   :titlesonly:

   data_import
   rotate
   run_ls_models
   ../_notebooks/figures
   