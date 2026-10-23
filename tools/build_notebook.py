"""Build a self-contained Colab notebook from the tested module and data."""
from pathlib import Path
import json
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source, hidden=False):
    cell = nbf.v4.new_code_cell(source.strip())
    if hidden:
        cell.metadata.update({"cellView": "form", "jupyter": {"source_hidden": True}})
    cells.append(cell)


md(r"""
# Membrane partition scales for helical peptides
**Grau et al., Science Advances 11, eads6804 (2025)** · [Paper](https://doi.org/10.1126/sciadv.ads6804) · [Open-access text and supplements](https://pmc.ncbi.nlm.nih.gov/articles/PMC11837994/)

Open this `.ipynb` in [Google Colab](https://colab.research.google.com/) using **File → Upload notebook**, then **Runtime → Run all**. The defaults run a sequence example and a clearly labeled synthetic PDB example. A CPU runtime is sufficient. All coefficients and scoring code are embedded; no repository checkout or supplementary-file upload is needed.

For your own helix, edit section 2 for a raw sequence or single FASTA record, or use sections 4–5 to upload a **membrane-oriented PDB**, inspect its chains, and select the helix. Tables, plots, coordinate selections, and a reproducibility manifest can be downloaded in section 7.

| Inputs | Implemented scales | Meaning |
|---|---|---|
| Sequence | In vitro Wat→Int and Wat→TM; in vivo Wat→Int and Wat→TM; derived Int→TM | Published per-residue coefficients; arbitrary-segment sums and means are additive scores |
| Sequence | HWvH at 4.5/7.5 Å; Wimley–White octanol/bilayer | Fixed comparison columns **as tabulated in Data S1**; HWvH columns are fixed-depth lookups, not its full position-dependent predictor |
| PDB coordinates | ΔG_PDB exposed, buried, and all residues | Signed-depth statistical potentials; optional per-residue exposure labels select among them |

**Units and interpretation.** Energies are kcal/mol and coordinates are Å. More negative scores favor the destination relative to that scale's reference. Int→TM = (Wat→TM) − (Wat→Int), so a negative difference favors TM over interface. The PDB values are statistical pseudo free energies. Summing them is an additive scoring assumption, not a calibrated probability of peptide insertion. Experimental and PDB scores retain their separate references.

**Calibration matters.** Experimental coefficients were fitted for substitutions at positions 11–13 of `WARYADWLFTXXXLLLLDLLALLV` (Eqs. 8–10). Summing over a different/full helix is a useful sequence descriptor but does not supply a validated absolute ΔG or three-state probability. Section 3 supplies the assay-specific calculation, counting only the substituted triplet.
""")

md("""
## 1. Load the implementation and published coefficients
Data are extracted from `ads6804_data_s1.xlsx` inside the supplied ZIP. No coefficients are refitted. The embedded JSON retains the spreadsheet cell ranges, full precision, and SHA-256 hashes.

- Data S1 lacks in vivo values for **A, M, Q, R**. An affected score is unavailable; missing contributions are never filled with zero or in vitro values.
- Supplementary Table 1 appears to interchange the rounded **E/P Wat→TM** entries relative to Data S1. This notebook uses Data S1: E = 0.174795, P = 0.249327.
- The Data S1 PDB formulas use **RT = 303 × 8.314 / 4184 = 0.6020893881 kcal/mol**, whereas Eq. 12 states **0.593**. The default reproduces the workbook; section 5 can instead rescale to the methods value.
- **Interpolation:** choose **Natural cubic splines** (default), **Piecewise linear interpolation**, or **Gaussian/sigmoid fits (to be implemented)** in section 5. Both implemented methods interpolate the Data S1 energy values independently for each amino acid and exposure category.
- **Original fitted model pending:** Rian Kormos will supply the sigmoid/Gaussian model used in Fig. 3; its coefficients were omitted from the supplied spreadsheet. Selecting that option raises an explanatory error until the original model is connected; it never silently substitutes another method.
""")
code('''#@title Install any missing dependencies and import plotting tools
import importlib.util
import subprocess
import sys
required = {"numpy": "numpy>=1.24", "pandas": "pandas>=2.0", "matplotlib": "matplotlib>=3.7", "Bio": "biopython>=1.81", "scipy": "scipy>=1.10"}
missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
import json
import hashlib
import platform
import importlib.metadata
from pathlib import Path
import zipfile
import matplotlib.pyplot as plt
from IPython.display import display
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
''')
code("#@title Embedded, auditable scoring implementation (expand to inspect)\n" + (ROOT / "membrane_scales.py").read_text(), True)
data = json.loads((ROOT / "data/scales.json").read_text())
code("#@title Embedded full-precision Data S1 coefficients (expand to inspect)\nDATA = json.loads(" +
     repr(json.dumps(data, ensure_ascii=False, separators=(",", ":"))) + ")\nscales = MembraneScales(DATA)\n"
     "print(DATA['citation'])\nprint('Workbook SHA-256:', DATA['workbook_sha256'])\n", True)
code('''#@title Original fitted model integration point (pending author-supplied model)
# Supply the original model here when available; no fit parameters are guessed.
# Adapter contract: f(population, aa, signed_z_array) -> energy_array in kcal/mol.
# Normalize the adapter's output to the workbook RT convention (0.6020893881)
# so the downstream explicit workbook/methods rescaling remains consistent.
# Supported populations are exposed, buried, all; each call receives one amino acid.
# The current API restricts fitted evaluation to [-39, 39] Å until its domain is verified.
ORIGINAL_FITTED_MODEL = None
scales = MembraneScales(DATA, fitted_model=ORIGINAL_FITTED_MODEL)
print("Original fit loaded" if ORIGINAL_FITTED_MODEL is not None else "Using Data S1 reference tables; original fit pending.")
''')

md("""
## 2. Score a sequence
Supply one helix in one-letter notation (lowercase and whitespace accepted). A single FASTA record is also accepted if editing the code. No alignment gaps, ambiguous residues, terminal modifications, or noncanonical amino acids are assumed.

The uncertainty column is an **approximate coefficient SE**, using `sqrt(sum(n_a² × SE_a²))`: repeated copies of a residue share the same fitted coefficient. Cross-coefficient covariances were not supplied; this is not a prediction interval. Int→TM uncertainty is left unavailable because covariance between the two fitted scales is unknown. The plot uses mean scores to make length dependence explicit.
""")
code('''#@title Sequence input and results
sequence_input = "WARYADWLFTDLPLLLLDLLALLV" #@param {type:"string"}
include_comparison_scales = True #@param {type:"boolean"}
sequence_summary = sequence_residues = None
sequence = normalize_sequence(sequence_input)
sequence_summary, sequence_residues = scales.sequence(sequence, include_comparison_scales)
print(f"{len(sequence)} residues: {sequence}")
display(sequence_summary.round(6))
display(sequence_residues.round(6))

available = sequence_summary[sequence_summary.status == "complete"]
sequence_figure, ax = plt.subplots(figsize=(9, max(3, len(available)*0.43)))
colors = ["#267c91" if "int_to_tm" not in k else "#b56e27" for k in available.scale]
ax.barh(available.label, available.mean_kcal_mol, color=colors)
ax.axvline(0, color="0.4", linewidth=0.8)
ax.invert_yaxis()
ax.set(xlabel="Mean additive score (kcal/mol per residue)", title="Sequence scores · each scale retains its own reference")
sequence_figure.tight_layout()
plt.show()
''')

md("""
## 3. Optional: calibrated bRc triplet calculation
The example `DLP` is the original bRc triplet. Enter a triplet or the entire **matching bRc scaffold**. This sums only positions 11–13, sets the aqueous-state energy to zero, and normalizes `exp(−ΔG/RT)` over Wat, Int, and TM. These are fitted model estimates for that experimental context. This section is independent of the arbitrary sequence in section 2.
""")
code('''#@title bRc assay-specific prediction
brc_triplet_or_scaffold = "DLP" #@param {type:"string"}
brc_assay = "vitro" #@param ["vitro", "vivo"]
scaffold_result = None
try:
    scaffold_result = scales.scaffold_prediction(brc_triplet_or_scaffold, brc_assay)
    display(scaffold_result.round(6))
except ValueError as exc:
    print(f"Assay prediction unavailable: {exc}")
''')

md(r"""
## 4. Load and inspect a PDB
Choose **demo**, **upload** (Colab's file picker), or **path** (a file already in the runtime). Upload exactly one PDB file. The demo is a synthetic, idealized Cα/Cβ helix centered at the origin; it is neither an experimental structure nor a complete all-atom model.

A generic PDB coordinate frame does not identify the membrane. Use a structure already positioned in the bilayer (for example, an OPM-oriented structure), or supply a known center and normal in section 5. The notebook does not position a helix by minimizing its score.
""")
code('''#@title PDB source and chain inventory
pdb_source = "demo" #@param ["demo", "upload", "path"]
pdb_path = "/content/helix.pdb" #@param {type:"string"}
pdb_summary = pdb_residues = coordinates = None
if pdb_source == "demo":
    pdb_text = demo_pdb()
    pdb_input_name = "synthetic_helix.pdb"
elif pdb_source == "upload":
    try:
        from google.colab import files
    except ImportError as exc:
        raise RuntimeError("Use pdb_source='path' outside Colab.") from exc
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError("Upload exactly one PDB file.")
    pdb_input_name, pdb_bytes = next(iter(uploaded.items()))
    pdb_text = pdb_bytes.decode("utf-8-sig")
elif pdb_source == "path":
    pdb_input_name = Path(pdb_path).name
    pdb_text = Path(pdb_path).read_text(encoding="utf-8-sig")
else:
    raise ValueError("Unknown PDB source")
display(pdb_inventory(pdb_text))
''')

md(r"""
## 5. Select a helix and define the membrane frame
Select **one continuous helix** by chain and inclusive residue IDs (e.g. `10`, `10A`, `-2`). Empty endpoints use the chain ends. `model_index` is zero-based order in the file; the inventory also shows the PDB MODEL serial. Helicity itself is not inferred.

The paper uses **Cβ**, or **Cα for Gly**, to define each residue's depth:
\[
z_i=(\mathbf r_i-\mathbf c)\cdot\frac{\mathbf n}{\|\mathbf n\|}.
\]
`center_xyz` is a point on the bilayer midplane; the normal points toward **positive z**, with the cytoplasmic side at negative z. The conventional oriented frame is center `0,0,0`, normal `0,0,1`. Signed depths are preserved: the two leaflets are not averaged. The structure is never automatically recentered, tilted, or rescaled to a chosen membrane thickness.

**Exposure:** exposed, buried, and pooled tables are reported separately at the same coordinates. For an isolated peptide, exposed is the natural default interpretation. You can provide a comma-separated label per selected residue to obtain an additional mixed score. Automatic burial assignment is not performed: the paper's assembly-dependent 8 Å alpha-sphere procedure cannot be replaced by an ordinary whole-protein convex hull or inferred reliably from an isolated helix.

**Interpolation:** select one of the three dropdown options below. **Natural cubic splines** (default) pass through all published bin-center energies and have zero second derivative at −39 and +39 Å (`CubicSpline(..., bc_type="natural")`). **Piecewise linear interpolation** connects adjacent bin-center energies with straight lines. **Gaussian/sigmoid fits (to be implemented)** is a placeholder for the original model and raises an explanatory error until available. After changing the dropdown, rerun this cell and the subsequent plotting/export cells.

**Domain:** both implemented interpolation methods use **−39 to +39 Å**. Default out-of-range behavior is an error; spline extrapolation is disabled. Explicit `clip` uses boundary values and marks every affected residue; it does not assume an aqueous zero outside the table. Highest-occupancy alternate atoms are selected (first on ties) and reported. Missing Cβ for a non-Gly residue (or missing Cα for Gly) is always an error; no atom substitution is made. Unknown residues are rejected.
""")
code('''#@title PDB selection and scoring parameters
chain_id = "A" #@param {type:"string"}
start_residue = "" #@param {type:"string"}
end_residue = "" #@param {type:"string"}
model_index = 0 #@param {type:"integer"}
center_xyz = "0,0,0" #@param {type:"string"}
normal_xyz = "0,0,1" #@param {type:"string"}
membrane_frame_confirmed = False #@param {type:"boolean"}
interpolation_method = "Natural cubic splines" #@param ["Natural cubic splines", "Piecewise linear interpolation", "Gaussian/sigmoid fits (to be implemented)"]
out_of_range = "error" #@param ["error", "clip"]
pdb_rt_convention = "workbook" #@param ["workbook", "methods"]
per_residue_exposure = "" #@param {type:"string"}

pdb_summary = pdb_residues = coordinates = None
depth_method = {
    "Natural cubic splines": "spline",
    "Piecewise linear interpolation": "linear",
    "Gaussian/sigmoid fits (to be implemented)": "fitted",
}[interpolation_method]
if depth_method == "fitted" and scales.fitted_model is None:
    raise ValueError("Gaussian/sigmoid fits are not yet implemented. Please select Natural cubic splines or Piecewise linear interpolation.")
if pdb_source != "demo" and not membrane_frame_confirmed:
    raise ValueError("Define/verify the membrane center and normal, then set membrane_frame_confirmed=True.")
center = tuple(float(x.strip()) for x in center_xyz.split(","))
normal = tuple(float(x.strip()) for x in normal_xyz.split(","))
coordinates = pdb_coordinates(pdb_text, chain_id, start_residue, end_residue, model_index)
exposure_labels = [x.strip().lower() for x in per_residue_exposure.split(",")] if per_residue_exposure.strip() else None
pdb_summary, pdb_residues = scales.score_pdb(coordinates, center, normal, depth_method, out_of_range, pdb_rt_convention, exposure_labels)
pdb_sequence = "".join(coordinates.aa)
print(f"Selected {len(coordinates)} residues: {pdb_sequence}")
print(f"Interpolation: {interpolation_method}")
print(f"Signed depth range: {pdb_residues.z_A.min():.3f} to {pdb_residues.z_A.max():.3f} Å")
display(pdb_summary.round(6))
display(pdb_residues.round(6))
print("Depth-independent scores for the SAME PDB-derived sequence:")
pdb_sequence_summary, pdb_sequence_residues = scales.sequence(pdb_sequence, include_comparison_scales)
display(pdb_sequence_summary.round(6))
''')
code('''#@title Inspect coordinates and per-residue scores
if pdb_residues is None:
    raise RuntimeError("Run PDB scoring successfully first.")
pdb_figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
positions = pdb_residues.position
axes[0].plot(positions, pdb_residues.z_A, "o-", color="#267c91", markersize=4)
axes[0].axhline(0, color="0.4", linewidth=0.8, label="bilayer midplane")
axes[0].axhspan(-15, 15, color="#e8c76a", alpha=0.18, label="±15 Å visual guide")
axes[0].set(ylabel="Signed depth (Å)", title="Input coordinates projected onto the supplied membrane normal")
axes[0].legend(fontsize=8)
for population, color in [("exposed", "#267c91"), ("buried", "#b56e27"), ("all", "#7e779c")]:
    axes[1].plot(positions, pdb_residues[f"{population}_kcal_mol"], "o-", label=population, color=color, markersize=3)
if "selected_kcal_mol" in pdb_residues:
    axes[1].plot(positions, pdb_residues.selected_kcal_mol, "k--", label="user labels")
axes[1].axhline(0, color="0.4", linewidth=0.8)
axes[1].set(ylabel="Residue score (kcal/mol)", xlabel="Selected PDB residue (chain : ID : amino acid)")
if len(positions) <= 60:
    axes[1].set_xticks(positions, [f"{r.chain}:{r.residue_id}:{r.aa}" for r in pdb_residues.itertuples()], rotation=90)
axes[1].legend()
pdb_figure.tight_layout()
plt.show()
''')

md("""
## 6. Inspect the underlying depth profiles
Dots are the published bin-center energies; curves use the **same interpolation selection as PDB scoring** in section 5, evaluated on a dense depth grid. The selected RT convention applies to both dots and curves. The spline and linear curves are not the sigmoid/Gaussian fits shown in Fig. 3.
""")
code('''#@title Depth profile viewer
profile_amino_acids = "WLDEK" #@param {type:"string"}
if pdb_residues is None:
    raise RuntimeError("Run PDB scoring successfully with the chosen interpolation method first.")
profile_aas = list(dict.fromkeys(normalize_sequence(profile_amino_acids)))
profile_figure, axes = plt.subplots(len(profile_aas), 1, figsize=(10, max(3, 2.2*len(profile_aas))), sharex=True, squeeze=False)
for aa, ax in zip(profile_aas, axes[:, 0]):
    for population, color in [("exposed", "#267c91"), ("buried", "#b56e27"), ("all", "#7e779c")]:
        grid = DATA["depth_scales"][population]["z"]
        dense_z = np.linspace(grid[0], grid[-1], 781)
        profile = scales.depth(aa*len(dense_z), dense_z, population, method=depth_method, rt=pdb_rt_convention)
        published = scales.depth(aa*len(grid), grid, population, method="linear", rt=pdb_rt_convention)
        ax.plot(dense_z, profile.energy_kcal_mol, label=population, color=color)
        ax.plot(grid, published.energy_kcal_mol, "o", markersize=2.5, color=color)
    ax.axhline(0, color="0.6", linewidth=0.7)
    ax.set(ylabel=f"{aa} (kcal/mol)")
axes[0, 0].legend(ncol=3)
axes[0, 0].set_title(interpolation_method)
axes[-1, 0].set_xlabel("Signed membrane depth (Å)")
profile_figure.tight_layout()
plt.show()
''')

md("""
## 7. Export results and reproducibility information
The ZIP contains CSV results (blank numeric values mean unavailable), PNG/SVG figures, all scale coefficients, and a JSON manifest with inputs, coordinate conventions, versions, and source hashes. Full uploaded PDB contents are not included; selected coordinates are in the residue table. Download is optional, so **Run all** completes without opening a download dialog.
""")
code('''#@title Save results (optionally download in Colab)
download_results = False #@param {type:"boolean"}
if sequence_summary is None or pdb_summary is None:
    raise RuntimeError("Run the sequence and PDB scoring cells successfully before exporting.")
output_dir = Path("membrane_scale_results")
output_dir.mkdir(exist_ok=True)
exported = []
tables = {"sequence_summary": sequence_summary, "sequence_residues": sequence_residues,
          "pdb_summary": pdb_summary, "pdb_residues": pdb_residues,
          "pdb_sequence_summary": pdb_sequence_summary, "pdb_sequence_residues": pdb_sequence_residues}
if scaffold_result is not None:
    tables["brc_scaffold_prediction"] = scaffold_result
for name, table in tables.items():
    path = output_dir / f"{name}.csv"
    table.to_csv(path, index=False)
    exported.append(path)
for name, figure in [("sequence", sequence_figure), ("pdb", pdb_figure), ("depth_profiles", profile_figure)]:
    for extension in ("png", "svg"):
        path = output_dir / f"{name}.{extension}"
        figure.savefig(path, bbox_inches="tight", dpi=180)
        exported.append(path)
manifest = {
    "citation": DATA["citation"], "doi": DATA["doi"], "workbook_sha256": DATA["workbook_sha256"],
    "sequence": sequence, "include_comparison_scales": include_comparison_scales,
    "brc_input": brc_triplet_or_scaffold, "brc_assay": brc_assay,
    "pdb_input_name": pdb_input_name, "pdb_source": pdb_source,
    "pdb_text_sha256": hashlib.sha256(pdb_text.encode("utf-8")).hexdigest(),
    "chain_id": chain_id, "start_residue": start_residue, "end_residue": end_residue,
    "model_index": model_index, "selected_sequence": pdb_sequence,
    "center_xyz_A": center, "positive_membrane_normal": normal,
    "membrane_frame_confirmed": membrane_frame_confirmed or pdb_source == "demo",
    "depth_method": depth_method, "out_of_range": out_of_range, "pdb_rt_convention": pdb_rt_convention,
    "interpolation_method": interpolation_method,
    "pdb_rt_kcal_mol": DATA["pdb_rt_" + pdb_rt_convention],
    "depth_atoms": "CB for non-Gly; CA for Gly; missing atoms are errors", "exposure_labels": exposure_labels,
    "original_fitted_model_loaded": ORIGINAL_FITTED_MODEL is not None,
    "units": {"energy": "kcal/mol", "coordinates": "angstrom"},
    "notes": DATA["notes"], "python": platform.python_version(),
    "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "matplotlib", "biopython", "scipy")}
}
for name, value in [("manifest.json", manifest), ("scales.json", DATA)]:
    path = output_dir / name
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    exported.append(path)
archive_path = Path("membrane_scale_results.zip")
with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in exported:
        archive.write(path, path.name)
print(f"Wrote {len(exported)} files to {archive_path.resolve()}")
if download_results:
    try:
        from google.colab import files
    except ImportError:
        print("Outside Colab: open the ZIP at the path above.")
    else:
        files.download(str(archive_path))
''')

md("""
### Provenance and scope
Source: Grau B., Kormos R., Bañó-Polo M., et al. *Sequence-dependent scale for translocon-mediated insertion of interfacial helices in membranes.* Science Advances **11**, eads6804 (2025), [doi:10.1126/sciadv.ads6804](https://doi.org/10.1126/sciadv.ads6804). Experimental coefficients: Data S1, `∆Gapp Per Residue!B3:M22`; PDB potentials: `∆G PDB!AW3:BR42`, `AW46:BR85`, `AW89:BR128`. PDB aqueous references use exposed propensities at bin centers −39, −37, −35, +35, +37, +39 Å for all three populations.

The notebook's additions are explicit interpolation, PDB input handling, additive whole-segment scores, and plotting/export. It does not reproduce UMAP embeddings, fit new experimental coefficients, infer helicity, assign a membrane orientation, or implement the original assembly burial classifier. No protonation, termini, helix-length, or hydrophobic-moment correction is added to the published coefficients. The article and supplied data are attributed to their authors and distributed under the article's [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/).
""")

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
    "colab": {"name": "membrane_scales_colab.ipynb", "provenance": [], "toc_visible": True},
})
for i, cell in enumerate(notebook.cells):
    cell.id = f"membrane-{i:02d}"
nbf.validate(notebook)
destination = ROOT / "membrane_scales_colab.ipynb"
nbf.write(notebook, destination)
print(f"Wrote {destination} ({len(cells)} cells)")
