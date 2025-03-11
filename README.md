# ProtLigNet Parser

A Python module to parse protein-ligand complexes from PDB files for machine learning-based optimization of protein-ligand interactions.

## Overview

This module provides tools to:
1. Parse PDB files and extract protein chains and ligands
2. Identify binding site residues within a specified distance cutoff
3. Generate modified structures for machine learning training
4. Create graph representations of protein-ligand interactions
5. Extract and visualize binding features

## Requirements

- Python 3.7+
- MDAnalysis
- NumPy
- Pandas
- NetworkX
- Matplotlib
- SciPy

Optional:
- RDKit (for enhanced ligand analysis)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/protlignet.git
cd protlignet

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### As a Command-Line Tool

```bash
python protlignet.py /path/to/protein_ligand.pdb --output results --binding-cutoff 5.0 --visualize
```

### As a Python Module

```python
from protlignet import ProtLigNetParser

# Initialize the parser
parser = ProtLigNetParser("protein_ligand.pdb", output_dir="results")

# Identify protein and ligand components
protein, ligands = parser.identify_components()

# Save components to separate PDB files
output_files = parser.save_components(prefix="complex1")

# Identify binding residues (within 5Å of ligand)
binding_residues = parser.identify_binding_residues()

# Generate files for machine learning
binding_site_files = parser.generate_binding_site_files()

# Create and visualize a binding interaction graph
binding_graph = parser.create_binding_graph()
parser.visualize_binding_graph("binding_graph.png")

# Extract features for machine learning
features = parser.extract_features()
```

## Key Features

### PDB Parsing and Component Extraction

```python
# Initialize with custom ligand identification
parser = ProtLigNetParser("complex.pdb")
protein, ligands = parser.identify_components(
    ligand_resnames=["ATP", "ADP", "NAD"],  # Specify ligands to look for
    ignore_water=True,                      # Ignore water molecules
    ignore_ions=True                        # Ignore common ions
)
```

### Binding Site Analysis

```python
# Identify residues within 4.5Å of any ligand atom
parser = ProtLigNetParser("complex.pdb", binding_cutoff=4.5)
binding_residues = parser.identify_binding_residues()
```

### Command-Line Options

```
usage: protlignet.py [-h] [--output OUTPUT] [--binding-cutoff BINDING_CUTOFF]
                     [--ligand-resnames LIGAND_RESNAMES [LIGAND_RESNAMES ...]]
                     [--visualize]
                     pdb_file

Parse and analyze protein-ligand complexes.

positional arguments:
  pdb_file              Path to PDB file

optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT, -o OUTPUT
                        Directory to save output files
  --binding-cutoff BINDING_CUTOFF, -b BINDING_CUTOFF
                        Distance cutoff (in Å) for defining binding site residues
  --ligand-resnames LIGAND_RESNAMES [LIGAND_RESNAMES ...], -l LIGAND_RESNAMES [LIGAND_RESNAMES ...]
                        Residue names to consider as ligands (optional)
  --visualize, -v       Visualize binding graph
```

## Output Files

The parser generates several files in the specified output directory:

- `{prefix}_protein.pdb`: The extracted protein structure
- `{prefix}_{ligand_name}_{index}.pdb`: Each extracted ligand
- `binding_site_metadata.csv`: Information about binding residues
- `binding_features.json`: Features for machine learning
- `binding_graph.png`: Visualization of protein-ligand interactions

## Example

```python
# Parse a protein-ATP complex
parser = ProtLigNetParser("protein_atp.pdb")
parser.identify_components(ligand_resnames=["ATP"])
parser.save_components()
parser.identify_binding_residues()
parser.generate_binding_site_files()
parser.create_binding_graph()
parser.visualize_binding_graph("atp_binding.png")
```

## Obtaining Sample PDB Files

You can download protein-ligand complex structures from the RCSB Protein Data Bank (https://www.rcsb.org/). Here are some examples of well-known protein-ligand complexes:

- 1ATP: ATP bound to an ATPase
- 3PGK: PGK with ATP
- 4XDJ: Protein kinase with inhibitor
- 1LVJ: Hemoglobin with heme
- 3M3X: PPAR-gamma with ligand

## Notes

- The parser automatically attempts to identify ligands based on common ligand names and size
- For better ligand detection, explicitly provide ligand residue names
- Residues are considered binding if any atom is within the specified cutoff distance
