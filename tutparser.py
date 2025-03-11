#!/usr/bin/env python
"""
Tutorial script demonstrating how to use the ProtLigNet parser module
with a sample PDB file from the RCSB PDB.
"""
import os
import sys
import logging
import requests
from pathlib import Path

# Import the ProtLigNetParser
from protlignet import ProtLigNetParser

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ProtLigNetTutorial")

def download_pdb(pdb_id, output_dir):
    """Download a PDB file from RCSB PDB if not already present."""
    pdb_path = Path(output_dir) / f"{pdb_id}.pdb"
    
    if pdb_path.exists():
        logger.info(f"Using existing PDB file: {pdb_path}")
        return str(pdb_path)
    
    logger.info(f"Downloading PDB file for {pdb_id}...")
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Create directory if it doesn't exist
        pdb_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save PDB file
        with open(pdb_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Downloaded PDB file to {pdb_path}")
        return str(pdb_path)
    
    except Exception as e:
        logger.error(f"Failed to download PDB file: {e}")
        return None

def main():
    # Create output directory
    output_dir = Path("tutorial_output")
    output_dir.mkdir(exist_ok=True)
    
    # Download a sample PDB file (1ATP: ATP bound to Ca-ATPase)
    pdb_id = "1ATP"
    pdb_path = download_pdb(pdb_id, output_dir)
    
    if not pdb_path:
        logger.error("Could not obtain PDB file. Exiting.")
        sys.exit(1)
    
    # Initialize the parser
    logger.info("Initializing ProtLigNetParser...")
    parser = ProtLigNetParser(pdb_path, output_dir=str(output_dir / pdb_id))
    
    # Step 1: Identify protein and ligand components
    logger.info("Step 1: Identifying protein and ligand components...")
    protein, ligands = parser.identify_components(
        ligand_resnames=["ATP"],  # Explicitly looking for ATP
        ignore_water=True,
        ignore_ions=True
    )
    
    if not ligands:
        logger.error("No ATP ligand found in the structure. Exiting.")
        sys.exit(1)
    
    logger.info(f"Found protein with {len(protein.residues)} residues")
    logger.info(f"Found {len(ligands)} ATP molecules")
    
    # Step 2: Save the components to separate PDB files
    logger.info("Step 2: Saving protein and ligand components to separate files...")
    output_files = parser.save_components(prefix=pdb_id)
    
    for name, path in output_files.items():
        logger.info(f"Saved {name} to {path}")
    
    # Step 3: Identify binding residues
    logger.info("Step 3: Identifying binding residues...")
    binding_residues = parser.identify_binding_residues()
    
    for lig_key, residues in binding_residues.items():
        logger.info(f"Found {len(residues)} binding residues for {lig_key}")
        # Print the first 5 binding residues
        for i, res in enumerate(residues[:5]):
            logger.info(f"  {i+1}. {res.resname}{res.resid}")
        if len(residues) > 5:
            logger.info(f"  ... and {len(residues)-5} more")
    
    # Step 4: Create binding graph
    logger.info("Step 4: Creating binding interaction graph...")
    binding_graph = parser.create_binding_graph()
    
    # Step 5: Visualize binding interactions
    logger.info("Step 5: Visualizing binding interactions...")
    graph_path = output_dir / pdb_id / "binding_graph.png"
    parser.visualize_binding_graph(str(graph_path))
    logger.info(f"Saved binding graph visualization to {graph_path}")
    
    logger.info("Tutorial completed successfully!")
    logger.info(f"Output files are in: {output_dir / pdb_id}")

if __name__ == "__main__":
    main()
