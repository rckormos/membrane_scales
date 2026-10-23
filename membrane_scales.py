"""Evaluate the scales in Grau et al. (2025), doi:10.1126/sciadv.ads6804.

Data are supplied explicitly, so this module can also be embedded in a notebook.
Energies are kcal/mol; coordinates and signed membrane depths are Å.
"""
from collections import Counter
from io import StringIO
import re
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

AA = "ACDEFGHIKLMNPQRSTVWY"
THREE_TO_ONE = dict(zip(
    "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(), AA
))
BRC = "WARYADWLFTDLPLLLLDLLALLV"


def normalize_sequence(text):
    """Accept one raw sequence or one FASTA record; reject gaps/unknown residues."""
    if not isinstance(text, str):
        raise ValueError("Sequence must be text.")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    headers = [i for i, line in enumerate(lines) if line.startswith(">")]
    if headers and headers != [0]:
        raise ValueError("Supply exactly one FASTA record, with its header first.")
    if headers:
        lines = lines[1:]
    seq = "".join("".join(lines).split()).upper()
    if not seq:
        raise ValueError("The sequence is empty.")
    invalid = sorted(set(seq) - set(AA))
    if invalid:
        raise ValueError(f"Unsupported sequence characters: {', '.join(invalid)}. Use the 20 canonical amino acids.")
    return seq


class MembraneScales:
    def __init__(self, data, fitted_model=None):
        self.data = data
        self.sequence_scales = data["sequence_scales"]
        self.depth_scales = data["depth_scales"]
        # Optional author-supplied callable: f(population, aa, signed_z_array).
        # Return one energy in kcal/mol per depth, on the workbook RT convention.
        self.fitted_model = fitted_model
        self._splines = {
            population: {aa: CubicSpline(table["z"], energies, bc_type="natural", extrapolate=False)
                         for aa, energies in table["energy"].items()}
            for population, table in self.depth_scales.items()
        }

    def sequence(self, text, include_comparisons=True):
        """Return summary and residue contributions; arbitrary-helix sums are scores.

        The reported SE approximation treats distinct fitted coefficients as
        independent, but repeated occurrences of the same coefficient are shared:
        variance = sum_a n_a**2 * SE_a**2. Regression covariance is unavailable.
        """
        seq = normalize_sequence(text)
        counts = Counter(seq)
        residues = pd.DataFrame({"position": range(1, len(seq)+1), "aa": list(seq)})
        rows = []
        for key, scale in self.sequence_scales.items():
            if scale["kind"] == "comparison" and not include_comparisons:
                continue
            values = np.array([np.nan if scale["values"][a] is None else scale["values"][a] for a in seq])
            residues[key] = values
            missing = sorted(a for a in counts if scale["values"][a] is None)
            complete = not missing
            se = np.nan
            if complete and scale["stderr"]:
                se = np.sqrt(sum((count * scale["stderr"][a])**2 for a, count in counts.items()))
            rows.append({"scale": key, "label": scale["label"], "n": len(seq),
                         "sum_kcal_mol": float(values.sum()) if complete else np.nan,
                         "mean_kcal_mol": float(values.mean()) if complete else np.nan,
                         "approx_sum_se": float(se),
                         "missing_aa": ",".join(missing),
                         "status": "complete" if complete else "unavailable (missing coefficients)"})
        for assay in ["vitro", "vivo"]:
            key = f"{assay}_int_to_tm"
            values = residues[f"{assay}_tm"] - residues[f"{assay}_int"]
            residues[key] = values
            missing = sorted(set(residues.loc[values.isna(), "aa"]))
            rows.append({"scale": key, "label": f"In {assay}: Int → TM (TM − Int)", "n": len(seq),
                         "sum_kcal_mol": float(values.sum()) if not missing else np.nan,
                         "mean_kcal_mol": float(values.mean()) if not missing else np.nan,
                         "approx_sum_se": np.nan, "missing_aa": ",".join(missing),
                         "status": "complete" if not missing else "unavailable (missing coefficients)"})
        return pd.DataFrame(rows), residues

    def scaffold_prediction(self, text, assay="vitro"):
        """Predict only the three substituted positions in the bRc host of Eq. 9.

        Accept a triplet or the exact 24-residue bRc background with positions
        11–13 varied. Do not sum the invariant background again.
        """
        if assay not in ("vitro", "vivo"):
            raise ValueError("assay must be vitro or vivo")
        seq = normalize_sequence(text)
        if len(seq) == 3:
            triplet = seq
        elif len(seq) == len(BRC) and seq[:10] == BRC[:10] and seq[13:] == BRC[13:]:
            triplet = seq[10:13]
        else:
            raise ValueError("Use a triplet or the bRc scaffold WARYADWLFTXXXLLLLDLLALLV.")
        summary, _ = self.sequence(triplet, include_comparisons=False)
        indexed = summary.set_index("scale")
        energies = np.array([0., indexed.loc[f"{assay}_int", "sum_kcal_mol"],
                             indexed.loc[f"{assay}_tm", "sum_kcal_mol"]])
        if not np.isfinite(energies).all():
            raise ValueError(f"The {assay} scale lacks coefficients for this triplet.")
        log_weights = -energies / self.data["experimental_rt"][assay]
        weights = np.exp(log_weights - log_weights.max())
        return pd.DataFrame({"state": ["Wat", "Int", "TM"], "triplet": triplet,
                             "delta_g_kcal_mol": energies, "model_probability": weights / weights.sum()})

    def depth(self, text, z, population="exposed", method="spline", bounds="error", rt="workbook"):
        """Evaluate signed depth with natural cubic splines (default) or linear interpolation.

        Natural splines interpolate energy values with zero second derivative
        at both endpoints; neither interpolation method is the Fig. 3 fit.
        Direct bin lookup remains available through the Python API.
        Bins span [-40, 40] Å and have centers -39,-37,...,39. At an internal
        bin edge use the higher-z bin; +40 belongs to the final bin.
        """
        seq = normalize_sequence(text)
        z = np.asarray(z, dtype=float)
        if z.shape != (len(seq),) or not np.isfinite(z).all():
            raise ValueError("Provide one finite signed depth per residue.")
        if population not in self.depth_scales:
            raise ValueError("population must be exposed, buried, or all")
        if method not in ("spline", "linear", "bin", "fitted") or bounds not in ("error", "clip"):
            raise ValueError("Use method='spline'/'linear'/'bin'/'fitted' and bounds='error'/'clip'.")
        if method == "fitted" and self.fitted_model is None:
            raise ValueError("Gaussian/sigmoid fits are not yet implemented: the original fitted model has not been supplied. "
                             "Choose natural cubic splines or piecewise linear interpolation.")
        if rt not in ("workbook", "methods"):
            raise ValueError("rt must be workbook or methods")
        table = self.depth_scales[population]
        grid = np.asarray(table["z"], dtype=float)
        # Keep an explicit empirical domain even for a fitted callable.
        lo, hi = (grid[0]-1, grid[-1]+1) if method == "bin" else (grid[0], grid[-1])
        outside = (z < lo) | (z > hi)
        if outside.any() and bounds == "error":
            positions = (np.flatnonzero(outside) + 1).tolist()
            raise ValueError(f"Depths outside [{lo:g}, {hi:g}] Å at sequence positions {positions}. "
                             "Check the membrane frame or explicitly choose bounds='clip'.")
        if outside.any():
            warnings.warn(f"Clipped {int(outside.sum())} depths to the published domain [{lo:g}, {hi:g}] Å.",
                          UserWarning, stacklevel=2)
        used = np.clip(z, lo, hi)
        if method == "fitted":
            values = np.empty(len(seq))
            letters = np.asarray(list(seq))
            for aa in set(seq):
                mask = letters == aa
                evaluated = np.asarray(self.fitted_model(population, aa, used[mask]), dtype=float)
                if evaluated.shape != used[mask].shape or not np.isfinite(evaluated).all():
                    raise ValueError(f"Fitted model must return a finite array of shape {used[mask].shape} for {aa}.")
                values[mask] = evaluated
        elif method == "spline":
            values = np.empty(len(seq))
            letters = np.asarray(list(seq))
            for aa in set(seq):
                mask = letters == aa
                values[mask] = self._splines[population][aa](used[mask])
        elif method == "linear":
            values = [np.interp(depth, grid, table["energy"][a]) for a, depth in zip(seq, used)]
        else:
            indices = np.minimum(np.floor((used - lo)/2).astype(int), len(grid)-1)
            values = [table["energy"][a][i] for a, i in zip(seq, indices)]
        factor = 1. if rt == "workbook" else self.data["pdb_rt_methods"] / self.data["pdb_rt_workbook"]
        return pd.DataFrame({"position": range(1, len(seq)+1), "aa": list(seq),
                             "z_A": z, "z_used_A": used, "clipped": outside,
                             "population": population, "energy_kcal_mol": np.asarray(values)*factor})

    def score_pdb(self, coordinates, center=(0., 0., 0.), normal=(0., 0., 1.),
                  method="spline", bounds="error", rt="workbook", exposure_labels=None):
        """Score selected coordinates in a user-defined membrane reference frame.

        Positive normal points away from the cytoplasmic side. Optional labels
        choose exposed/buried/all per residue; no exposure is inferred here.
        """
        center = np.asarray(center, dtype=float)
        normal = np.asarray(normal, dtype=float)
        if center.shape != (3,) or normal.shape != (3,) or not np.isfinite([center, normal]).all():
            raise ValueError("Membrane center and normal must each be three finite numbers.")
        norm = np.linalg.norm(normal)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("Membrane normal must be nonzero and finite.")
        if len(coordinates) == 0:
            raise ValueError("The coordinate selection is empty.")
        result = coordinates.copy().reset_index(drop=True)
        xyz = result[["x_A", "y_A", "z_cart_A"]].to_numpy(dtype=float)
        if not np.isfinite(xyz).all():
            raise ValueError("Coordinates must be finite.")
        depths = (xyz - center) @ (normal / norm)
        seq = "".join(result["aa"])
        summary = []
        for population in ("exposed", "buried", "all"):
            scores = self.depth(seq, depths, population, method, bounds, rt)
            result[f"{population}_kcal_mol"] = scores["energy_kcal_mol"].to_numpy()
            summary.append({"population": population, "n": len(result),
                            "sum_kcal_mol": float(scores.energy_kcal_mol.sum()),
                            "mean_kcal_mol": float(scores.energy_kcal_mol.mean()),
                            "method": method, "rt": rt})
        result["z_A"] = depths
        result["z_used_A"] = scores["z_used_A"].to_numpy()
        result["clipped"] = scores["clipped"].to_numpy()
        if exposure_labels is not None:
            labels = list(exposure_labels)
            if len(labels) != len(result) or not set(labels) <= set(self.depth_scales):
                raise ValueError("Supply one label (exposed, buried, or all) per selected residue.")
            result["exposure_label"] = labels
            result["selected_kcal_mol"] = [result.loc[i, f"{label}_kcal_mol"] for i, label in enumerate(labels)]
            summary.append({"population": "user labels", "n": len(result),
                            "sum_kcal_mol": float(result.selected_kcal_mol.sum()),
                            "mean_kcal_mol": float(result.selected_kcal_mol.mean()),
                            "method": method, "rt": rt})
        return pd.DataFrame(summary), result


def _structure(text):
    if not isinstance(text, str) or not any(line.startswith(("ATOM  ", "HETATM")) for line in text.splitlines()):
        raise ValueError("Supply PDB-format text containing ATOM/HETATM records (not mmCIF).")
    # Strict parsing rejects malformed/duplicate records instead of silently dropping them.
    return PDBParser(PERMISSIVE=False, QUIET=False).get_structure("input", StringIO(text))


def _protein_residues(chain):
    return [r for r in chain if r.id[0] == " " or is_aa(r, standard=False)
            or ("CA" in r and "N" in r and "C" in r)]


def _residue_id(residue):
    return f"{residue.id[1]}{residue.id[2].strip()}"


def pdb_inventory(text):
    rows = []
    for ordinal, model in enumerate(_structure(text)):
        for chain in model:
            residues = _protein_residues(chain)
            if residues:
                rows.append({"model_index": ordinal, "pdb_model_serial": model.serial_num,
                             "chain": chain.id, "first": _residue_id(residues[0]),
                             "last": _residue_id(residues[-1]), "residues": len(residues),
                             "sequence": "".join(THREE_TO_ONE.get(r.resname, "X") for r in residues)})
    return pd.DataFrame(rows)


def pdb_coordinates(text, chain_id="A", start=None, end=None, model_index=0):
    """Decode one contiguous helix selection, preserving numbering/insertion codes.

    start/end are inclusive PDB residue IDs (e.g. '10', '10A', '-2'). None
    selects the respective chain end. The user identifies the helix; this
    function does not infer secondary structure or membrane orientation.
    Bio.PDB selects the highest-occupancy alternate atom (first on a tie).
    """
    models = list(_structure(text))
    if not isinstance(model_index, int) or not 0 <= model_index < len(models):
        raise ValueError(f"model_index must be in 0..{len(models)-1} (zero-based file order).")
    model = models[model_index]
    if chain_id not in model:
        raise ValueError(f"Chain {chain_id!r} not found. Available: {[c.id for c in model]}")
    residues = _protein_residues(model[chain_id])
    ids = [_residue_id(r) for r in residues]
    if len(ids) != len(set(ids)):
        raise ValueError("Ambiguous duplicate residue IDs within the selected chain.")
    if not ids:
        raise ValueError("No peptide residues found in this chain.")
    def endpoint(value, default):
        if value is None or str(value).strip() == "":
            return default
        token = str(value).strip()
        if not re.fullmatch(r"-?\d+[A-Za-z]?", token) or token not in ids:
            raise ValueError(f"Residue endpoint {token!r} is not present in chain {chain_id!r}.")
        return ids.index(token)
    first, last = endpoint(start, 0), endpoint(end, len(ids)-1)
    if first > last:
        raise ValueError("The start residue occurs after the end residue in the PDB.")
    selected = residues[first:last+1]
    rows = []
    for i, residue in enumerate(selected):
        rid = _residue_id(residue)
        if residue.is_disordered() == 2:
            raise ValueError(f"Residue {rid} has alternate residue identities; resolve these first.")
        if residue.resname not in THREE_TO_ONE:
            raise ValueError(f"Unsupported residue {residue.resname} at {chain_id}:{rid}; no implicit mutation is applied.")
        aa = THREE_TO_ONE[residue.resname]
        atom_name = "CA" if aa == "G" else "CB"
        if atom_name not in residue:
            raise ValueError(f"Missing {atom_name} at {chain_id}:{rid}. Complete the structure before scoring; no atom substitution is allowed.")
        atom = residue[atom_name]
        xyz = atom.coord.astype(float)
        if not np.isfinite(xyz).all():
            raise ValueError(f"Nonfinite coordinates at {chain_id}:{rid}.")
        if i and "CA" in residue and "CA" in selected[i-1]:
            distance = np.linalg.norm(residue["CA"].coord - selected[i-1]["CA"].coord)
            if distance > 4.5:
                raise ValueError(f"Cα gap of {distance:.2f} Å before {rid}; select one continuous helix.")
        rows.append({"position": i+1, "model_index": model_index, "chain": chain_id,
                     "residue_id": rid, "resname": residue.resname, "aa": aa,
                     "depth_atom": atom_name, "altloc": atom.get_altloc().strip(),
                     "occupancy": atom.get_occupancy(),
                     "x_A": xyz[0], "y_A": xyz[1], "z_cart_A": xyz[2]})
    return pd.DataFrame(rows)


def demo_pdb(sequence=BRC):
    """Synthetic idealized Cα/Cβ helix for a reproducible input-path demonstration.

    This is not an experimental structure or a complete all-atom model.
    """
    seq = normalize_sequence(sequence)
    one_to_three = {v: k for k, v in THREE_TO_ONE.items()}
    lines = ["REMARK   1 SYNTHETIC CA/CB HELIX; MEMBRANE CENTER 0,0,0; NORMAL 0,0,1"]
    serial = 0
    for i, aa in enumerate(seq):
        angle = np.deg2rad(i*100.)
        for name, radius in [("CA", 2.3)] + ([] if aa == "G" else [("CB", 3.8)]):
            serial += 1
            x, y, z = radius*np.cos(angle), radius*np.sin(angle), (i-(len(seq)-1)/2)*1.5
            lines.append(f"ATOM  {serial:5d} {name:^4s} {one_to_three[aa]:3s} A{i+1:4d}    "
                         f"{x:8.3f}{y:8.3f}{z:8.3f}{1.:6.2f}{0.:6.2f}          C  ")
    return "\n".join(lines + ["TER   ", "END   "]) + "\n"
