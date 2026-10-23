"""Execute the standalone notebook in an empty temporary working directory."""
import argparse
import json
from pathlib import Path
import tempfile
import zipfile

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-outputs", action="store_true")
    args = parser.parse_args()
    path = ROOT / "membrane_scales_colab.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    with tempfile.TemporaryDirectory(prefix="membrane-notebook-") as workdir:
        NotebookClient(notebook, timeout=180, kernel_name="python3",
                       resources={"metadata": {"path": workdir}}).execute()
        archive = Path(workdir) / "membrane_scale_results.zip"
        with zipfile.ZipFile(archive) as zipped:
            manifest = json.loads(zipped.read("manifest.json"))
            assert manifest["selected_sequence"] == "WARYADWLFTDLPLLLLDLLALLV"
            assert manifest["pdb_source"] == "demo"
            assert manifest["depth_method"] == "spline"
            assert manifest["interpolation_method"] == "Natural cubic splines"
            assert "scipy" in manifest["packages"]
            assert len(zipped.namelist()) == 15
            assert "pdb_residues.csv" in zipped.namelist()
            assert "depth_profiles.svg" in zipped.namelist()
        print(f"Executed {sum(c.cell_type == 'code' for c in notebook.cells)} code cells; verified 15-file result archive.")
    if args.save_outputs:
        # Runtime-specific temporary paths in outputs are not useful in the deliverable.
        for cell in notebook.cells:
            for output in cell.get("outputs", []):
                if output.get("output_type") == "stream":
                    output["text"] = output["text"].replace(workdir + "/", "")
        nbformat.write(notebook, path)
        print(f"Saved example outputs in {path}")


if __name__ == "__main__":
    main()
