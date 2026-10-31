# Streamline filters

| script | purpose |
|---|---|
| `msp_fit.py`, `msp_fit_finite.py` | fit a full midsagittal plane x = XC + AY·Y + AZ·Z by mirror symmetry (NCC); the finite version sanitises NaN voxels first |
| `prune_midline_msp.py` | drop streamlines that cross the fitted (tilted) midline, keeping SIFT2 weights aligned |
| `prune_hdp_len_tort.py` | drop looping outliers by arc length and tortuosity (arc / end-to-end distance); thresholds chosen from each bundle's own distribution |
| `prune_hdp_physio.py` | physiological constraints for the hyperdirect pathway (cortex → internal capsule → STN) |
