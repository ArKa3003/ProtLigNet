import os
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
# Structural biology libraries
import MDAnalysis as mda
from scipy.spatial.distance import cdist

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ProtLigProcessor")

class ProtLigProcessor:
    """
    A class for processing protein-ligand complexes, identifying binding residues,
    and generating modified structures.
    
    This processor focuses on:
    1. Parsing protein-ligand complexes from PDB files
    2. Separating and retaining protein chains and ligands
    3. Identifying ligand-binding amino acids (within 5Å of any ligand atom)
    4. Generating two PDB files for each binding residue:
       - One file with just the ligand
       - One file with the protein where the identified binding residue has its 
         side chain stripped (keeping only backbone atoms)
    """

    def __init__(self, pdb_path: str, output_dir: str = "output", binding_cutoff: float = 5.0):
        """
        Initialize the processor with a PDB file.
        
        Args:
            pdb_path: Path to the PDB file
            output_dir: Directory to save output files
            binding_cutoff: Distance cutoff (in Å) for defining binding site residues
        """
        self.pdb_path = pdb_path
        self.output_dir = Path(output_dir)
        self.binding_cutoff = binding_cutoff
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Initialize data structures
        self.universe = None
        self.protein = None
        self.ligands = []
        self.binding_residues = {}  # Dictionary mapping ligand ID to binding residues
        # Load the PDB file
        self._load_pdb()
        logger.info(f"Initialized ProtLigProcessor with {pdb_path}")

    def _load_pdb(self):
        """Load the PDB file into an MDAnalysis universe."""
        try:
            self.universe = mda.Universe(self.pdb_path)
            logger.info(f"Successfully loaded PDB file with {len(self.universe.atoms)} atoms")
        except Exception as e:
            logger.error(f"Failed to load PDB file: {e}")
            raise

    def identify_components(self, protein_segids: Optional[List[str]] = None,
                           ligand_resnames: Optional[List[str]] = None,
                           ignore_water: bool = True,
                           ignore_ions: bool = True):
        """
        Identify protein chains and ligands in the structure.
        
        Args:
            protein_segids: List of segment IDs to consider as protein
            ligand_resnames: List of residue names to consider as ligands
            ignore_water: Whether to ignore water molecules
            ignore_ions: Whether to ignore common ions
            
        Returns:
            Tuple containing protein and ligand AtomGroups
        """
        # Common ions to ignore
        ion_residues = {'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'FE', 'MN', 'CO', 'NI', 'CU'}
        # Common ligand residues (if not specified)
        common_ligands = {
            'ADP', 'ATP', 'GTP', 'FMN', 'FAD', 'HEM', 'NAD', 'NAP', 'SAM', 'PLP',
            'COA', 'ACO', 'GDP', 'AMP', 'CMP', 'UMP', 'TMP', 'UDP', 'CPT', 'TTP', 'BTN',
            'HEC', 'SUC', 'FUC', 'MAL', 'GAL', 'MAN', 'BMA', 'WHO'
        }

        # Find protein chains
        protein_select = "protein"
        if protein_segids:
            segid_str = " or ".join([f"segid {segid}" for segid in protein_segids])
            protein_select = f"({protein_select}) and ({segid_str})"
        self.protein = self.universe.select_atoms(protein_select)
        logger.info(f"Identified protein with {len(self.protein.residues)} residues")

        # Find ligands
        exclude = []
        if ignore_water:
            exclude.append("resname HOH SOL WAT TIP3")
        if ignore_ions:
            ion_str = " ".join(ion_residues)
            exclude.append(f"resname {ion_str}")
        exclude_str = " and not ".join([""] + exclude)

        if ligand_resnames:
            ligand_str = " ".join(ligand_resnames)
            ligand_select = f"resname {ligand_str}"
        else:
            # Try to identify ligands automatically
            # First, select all HETATM records that are not protein, water, or ions
            ligand_select = f"(not protein){exclude_str}"
            # Add common ligands
            ligand_select += f" or resname {' '.join(common_ligands)}"

        potential_ligands = self.universe.select_atoms(ligand_select)
        
        # Group atoms by residue
        ligand_residues = potential_ligands.residues
        
        # Filter out small molecules (likely ions or artifacts)
        for residue in ligand_residues:
            if len(residue.atoms) >= 6:  # Arbitrary cutoff to filter out small molecules
                self.ligands.append(residue.atoms)
                logger.info(f"Identified ligand: {residue.resname} {residue.resid} with {len(residue.atoms)} atoms")

        if not self.ligands:
            logger.warning("No ligands found in the structure.")
        
        return self.protein, self.ligands

    def save_components(self, prefix: Optional[str] = None):
        """
        Save protein and ligand components to separate PDB files.
        
        Args:
            prefix: Optional prefix for output files
            
        Returns:
            Dictionary mapping component names to output file paths
        """
        output_files = {}
        if prefix is None:
            prefix = Path(self.pdb_path).stem
        
        # Save protein
        protein_path = self.output_dir / f"{prefix}_protein.pdb"
        self.protein.write(str(protein_path))
        output_files["protein"] = protein_path
        logger.info(f"Saved protein to {protein_path}")
        
        # Save ligands
        for i, ligand in enumerate(self.ligands):
            resname = ligand.residues[0].resname if ligand.residues else f"LIG{i+1}"
            ligand_path = self.output_dir / f"{prefix}_{resname}_{i+1}.pdb"
            ligand.write(str(ligand_path))
            output_files[f"ligand_{resname}_{i+1}"] = ligand_path
            logger.info(f"Saved ligand {resname} to {ligand_path}")
        
        return output_files

    def identify_binding_residues(self):
        """
        Identify protein residues within the binding cutoff distance of any ligand atom.
        
        Returns:
            Dictionary mapping ligand indices to lists of binding residues
        """
        if not self.protein or not self.ligands:
            logger.error("Protein or ligands not identified. Run identify_components() first.")
            return None
        
        self.binding_residues = {}
        for i, ligand in enumerate(self.ligands):
            # Get ligand residue name
            lig_name = ligand.residues[0].resname if ligand.residues else f"LIG{i+1}"
            key = f"{lig_name}_{i+1}"
            
            # Calculate distances between ligand atoms and protein residues
            binding_res_indices = set()
            
            # Use efficient distance calculation
            ligand_coords = ligand.positions
            protein_residues = self.protein.residues
            
            # For each protein residue, calculate minimum distance to any ligand atom
            for res_idx, residue in enumerate(protein_residues):
                res_coords = residue.atoms.positions
                # Calculate minimum distance between residue atoms and ligand atoms
                distances_matrix = cdist(res_coords, ligand_coords)
                min_distance = np.min(distances_matrix)
                
                if min_distance <= self.binding_cutoff:
                    binding_res_indices.add(res_idx)
            
            # Convert indices to actual residue objects
            binding_res = [protein_residues[idx] for idx in binding_res_indices]
            self.binding_residues[key] = binding_res
            logger.info(f"Identified {len(binding_res)} binding residues for ligand {key}")
        
        return self.binding_residues

    def generate_binding_site_files(self):
        """
        Generate PDB files for each binding residue:
        1. One file with the ligand only
        2. One file with the protein and the binding residue modified (side chain stripped)
        
        Returns:
            Dictionary mapping binding residues to output file paths
        """
        if not self.binding_residues:
            logger.error("Binding residues not identified. Run identify_binding_residues() first.")
            return None
        
        output_files = {}
        for lig_key, binding_res in self.binding_residues.items():
            ligand_idx = int(lig_key.split('_')[-1]) - 1
            ligand = self.ligands[ligand_idx]
            
            # Create directory for this ligand
            ligand_dir = self.output_dir / lig_key
            ligand_dir.mkdir(exist_ok=True)
            
            # Save ligand file (just once for this ligand)
            ligand_path = ligand_dir / f"{lig_key}.pdb"
            ligand.write(str(ligand_path))
            
            # For each binding residue, create modified protein
            for res in binding_res:
                res_id = f"{res.resname}_{res.resid}"
                output_files[f"{lig_key}_{res_id}"] = {}
                
                # Reference to the ligand file
                output_files[f"{lig_key}_{res_id}"]["ligand"] = ligand_path
                
                # The selection logic:
                # 1. Keep all protein atoms that are NOT part of this residue
                # 2. For this residue, keep ONLY backbone atoms (N, CA, C, O)
                backbone_atoms = ["N", "CA", "C", "O"]
                
                # Select all protein atoms except this residue
                other_residues = f"protein and not resid {res.resid}"
                
                # Select only backbone atoms for this residue
                this_residue_backbone = f"resid {res.resid} and name {' '.join(backbone_atoms)}"
                
                # Combine the selections
                selection = f"({other_residues}) or ({this_residue_backbone})"
                
                # Select the modified protein
                modified_protein = self.universe.select_atoms(selection)
                
                # Save modified protein
                modified_path = ligand_dir / f"{lig_key}_{res_id}_modified.pdb"
                modified_protein.write(str(modified_path))
                output_files[f"{lig_key}_{res_id}"]["modified_protein"] = modified_path
                
                logger.info(f"Generated modified protein for binding residue {res_id}")
    
        # Save metadata to CSV
        metadata = []
        for key, files in output_files.items():
            lig_key, res_id = key.split('_', 1)
            row = {
                "ligand": lig_key,
                "residue": res_id,
                "ligand_path": files["ligand"],
                "modified_protein_path": files.get("modified_protein", "")
            }
            metadata.append(row)
        
        # Save metadata
        if metadata:
            metadata_path = self.output_dir / "binding_site_metadata.csv"
            pd.DataFrame(metadata).to_csv(metadata_path, index=False)
            logger.info(f"Saved binding site metadata to {metadata_path}")
        
        return output_files

def main():
    """
    Main function to demonstrate the use of the ProtLigProcessor class.
    This function provides a command-line interface to the processor.
    """
    parser = argparse.ArgumentParser(description='Process protein-ligand complexes.')
    parser.add_argument('pdb_file', type=str, help='Path to PDB file')
    parser.add_argument('--output', '-o', type=str, default='output', 
                        help='Directory to save output files')
    parser.add_argument('--binding-cutoff', '-b', type=float, default=5.0,
                        help='Distance cutoff (in Å) for defining binding site residues')
    parser.add_argument('--ligand-resnames', '-l', type=str, nargs='+',
                        help='Residue names to consider as ligands (optional)')
    
    args = parser.parse_args()
    
    # Initialize the processor
    processor = ProtLigProcessor(args.pdb_file, args.output, args.binding_cutoff)
    
    # Identify components
    protein, ligands = processor.identify_components(ligand_resnames=args.ligand_resnames)
    
    # Print summary
    print(f"Found protein with {len(protein.residues)} residues")
    print(f"Found {len(ligands)} ligand(s)")
    
    # Save components
    output_files = processor.save_components()
    print(f"Saved {len(output_files)} component files to {args.output}")
    
    # Identify binding residues
    if ligands:
        binding_residues = processor.identify_binding_residues()
        for lig_key, residues in binding_residues.items():
            print(f"Found {len(residues)} binding residues for ligand {lig_key}")
        
        # Generate modified PDB files
        binding_files = processor.generate_binding_site_files()
        print(f"Generated files for {len(binding_files)} binding sites")
    
    return processor


if __name__ == "__main__":
    main()
