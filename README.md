# Membrane partition scales — Grau et al. (2025)

Open **[membrane_scales_colab.ipynb](membrane_scales_colab.ipynb)** in
[Google Colab](https://colab.research.google.com/) using **File → Upload notebook**.
Run all cells for a working sequence and synthetic PDB example. No GPU, repository
checkout, or supplementary-file upload is needed: the notebook embeds its code and
full-precision coefficients.

For your data, change the sequence form, or upload a membrane-oriented PDB, inspect
the chain inventory, and select a helix by chain and inclusive residue IDs.
The notebook produces per-residue scores, sums and means, plots, and a downloadable
ZIP with CSVs, PNG/SVG figures, coefficients, and a reproducibility manifest.

## Implemented scales

- In vitro and in vivo Wat→Int and Wat→TM, plus the derived Int→TM difference.
- The four comparison columns tabulated in Data S1: HWvH at 4.5/7.5 Å and
  Wimley–White octanol/bilayer. The HWvH columns are fixed-depth lookups, not a
  reproduction of the complete Hessa position-dependent predictor.
- PDB statistical potentials for exposed, buried, and pooled residues. The notebook
  interpolation dropdown offers **Natural cubic splines** (default),
  **Piecewise linear interpolation**, and **Gaussian/sigmoid fits (to be implemented)**.
  Plots and exported results use the selected method.
- A separate bRc triplet calculation reproduces the calibration context of
  Eqs. 8–10 and converts those energies to model state probabilities.

**Original fitted depth model pending:** Rian Kormos will supply the model used
for Fig. 3, whose parameters are absent from Data S1. The notebook currently
provides a labeled table-based reference and a callable integration point.
Selecting `fitted` without supplying the model raises an error. No coefficients
have been guessed or refitted. The adapter contract is
`f(population, amino_acid, signed_depth_array) -> kcal_per_mol_array`;
its normalization/domain must be checked when the original model arrives.

## Scientific conventions

- Use **Cβ for non-Gly and Cα for Gly**, strictly. Missing required atoms are errors.
- Signed depth is `(coordinate − center) · normalized_normal`, in Å. The membrane
  midplane is zero, with negative depth toward the cytoplasm. No automatic
  centering, orientation fitting, leaflet averaging, or thickness rescaling occurs.
- Supply an already oriented PDB or known center/normal. A generic PDB does not
  identify a membrane frame. The selected segment is assumed to be a helix;
  secondary structure is not inferred.
- Exposure classes are evaluated separately; optional per-residue labels provide
  a mixed score. The paper's assembly-dependent alpha-sphere classifier is not
  implemented or approximated by a simple convex hull.
- Natural cubic splines and linear interpolation use energy values at bin centers
  −39 to +39 Å. Splines have zero second derivative at both endpoints and do not
  extrapolate. Out-of-domain values raise unless clipping is explicitly enabled.
  Direct bin lookup (−40 to +40 Å) remains available in the Python API.
- Preserve missing in vivo coefficients (A, M, Q, R). An affected aggregate is
  unavailable rather than a misleading partial sum.
- Arbitrary-helix sums of the experimental coefficients are additive descriptors,
  not calibrated absolute transfer energies. Only the substituted triplet is
  counted for the bRc assay calculation.

Data S1 is the numerical source of truth. Its E/P in vitro TM coefficients are
0.174795/0.249327; rounded Supplementary Table 1 appears to interchange them.
PDB energies default to the workbook's `RT = 303 × 8.314 / 4184`, with an explicit
option for the main text's `RT = 0.593` convention. Reported coefficient-error
approximations account for repeated use of a fitted coefficient; the unavailable
regression covariances prevent full uncertainty propagation.

## Development and validation

```bash
python -m pip install -r requirements-dev.txt
python tools/extract_scales.py
python tools/build_notebook.py
python -m pytest -q
python tools/execute_notebook.py
```

The extraction script reads the supplied ZIP; it does not modify the source files.
Tests compare all coefficients and all 2,400 depth values to the workbook,
independently reconstruct reverse-Boltzmann energies, verify triplet calculations,
and exercise PDB selections, alternate atoms, missing atoms, signed coordinates,
and coordinate-frame invariance. Notebook execution uses its embedded code/data
from a temporary directory so it cannot accidentally depend on local imports.

Files:

- `membrane_scales.py`: reusable numerical and PDB-input functions.
- `data/scales.json`: extracted values, source-cell ranges, and source hashes.
- `tools/build_notebook.py`: rebuilds the self-contained notebook from those files.
- `tests/test_scales.py`: numerical and input-handling checks.

Source: Grau B., Kormos R., Bañó-Polo M., et al. *Sequence-dependent scale for
translocon-mediated insertion of interfacial helices in membranes.*
Science Advances **11**, eads6804 (2025).
[doi:10.1126/sciadv.ads6804](https://doi.org/10.1126/sciadv.ads6804).
The article and supplied data are attributed to their authors under the article's
[CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/).
