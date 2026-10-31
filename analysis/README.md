# Analysis scripts

Electrode-contact-to-tract distance measures and their statistics against
clinical improvement. Inputs are CSV/XLSX tables produced by the measurement
steps; no clinical data are stored in this repository.

- `dist_to_bundle_core.py`, `compare_prent_distance.py`: nearest-fibre distance
  saturates at 0 inside a bundle, so distances to the bundle core are computed too.
- `hdp_bootstrap_cum_muc6.py`: patient-level cluster bootstrap CIs and block
  permutation p-values (two hemispheres per patient are not independent).
- `hdp_do_ben_phep_do.py`, `cst_dcore_measurement_error.py`: robustness of the
  distance measure to tractography re-runs, subsampling and localisation error.
- `steiger_specificity_corrected.py`: Steiger tests for whether one tract
  correlates with outcome more strongly than another (anatomical specificity).
- `_xlsx_min.py`: minimal standard-library `.xlsx` reader.
