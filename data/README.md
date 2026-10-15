# Data

The real survey data **are not redistributed in this repository**. Download them from the original source and
check the licence stated there:

* Dataset (Mendeley Data): *Social Commerce Adoption Factors among Gen Z University Students in Vietnam*,
  DOI [10.17632/g59k85hrd5](https://doi.org/10.17632/g59k85hrd5) (version 2, 21 Oct 2025).
* Data article: *Dataset on social and behavioral factors influencing social commerce adoption among Gen Z
  university students in Vietnam*, Data in Brief (2026), PMCID PMC12907679.

Place the CSV at `data/S-COMMERCE_GenZ_UniversityStudent_757_DIB.csv` (any path works with `--data`).
It must have 757 rows and the 31 columns `Gender, Job, Income, Area, Frequently` plus the Likert items
`PU1-4, PEU1-4, FSC1-3, SP1-4, TP1-3, IB1-4, AUB1-4`.

## `synthetic_demo.csv`

A **synthetic** file with the same schema (400 rows, random latent-factor draws, produced by
`scripts/make_synthetic_data.py`). It exists only so that the pipeline, tests and CI can run without the real
data. Never report results obtained on it.
