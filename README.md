# Backchannel vs. Turn Initiations: ROI Analysis

This repository contains Python and R scripts used for the region-of-interest (ROI) analysis of functional MRI data comparing backchannel initiations (BI) and turn initiations (TI) during conversation.

- `extractmshbmbetas.py` — Extracts run-level BI and TI beta estimates from individualized MSHBM network masks.
- `mshbmRoiAnalysisScript.R` — Performs the statistical ROI analysis using linear mixed-effects modelling.

## Requirements

**Python:** NumPy, SciPy, NiBabel, Nilearn.

**R:** lme4, lmerTest, emmeans, dplyr.

The extraction script requires SPM first-level beta images, `SPM.mat` files, and subject-specific MSHBM surface masks. Local input and output paths must be configured before execution.
