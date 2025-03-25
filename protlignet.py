import os
import logging
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional, Union, Any
# Structural biology libraries
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from scipy.spatial.distance import cdist
# For fingerprinting ligands
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    from rdkit.Chem.Draw import IPythonConsole
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    logging.warning("RDKit not found. Ligand analysis will be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ProtLigNet")

# Define chemical properties
# Hydrophobicity scale (Kyte & Doolittle)
HYDROPHOBICITY = {
    'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5, 'MET': 1.9, 'ALA': 1.8,
    'GLY': -0.4, 'THR': -0.7, 'SER': -0.8, 'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6,
    'HIS': -3.2, 'GLU': -3.5, 'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5
}

# Electrostatic properties
CHARGE = {
    'ARG': 1, 'LYS': 1, 'HIS': 0.1, 'ASP': -1, 'GLU': -1,
    'SER': 0, 'THR': 0, 'ASN': 0, 'GLN': 0, 'CYS': 0,
    'GLY': 0, 'PRO': 0, 'ALA': 0, 'VAL': 0, 'ILE': 0,
    'LEU': 0, 'MET': 0, 'PHE': 0, 'TYR': 0, 'TRP': 0
}

# Hydrogen bond potential
H_BOND = {
    'SER': 2, 'THR': 2, 'ASN': 4, 'GLN': 4, 'TYR': 2, 'TRP': 1,
    'HIS': 4, 'LYS': 2, 'ARG': 6, 'ASP': 4, 'GLU': 4,
    'GLY': 0, 'PRO': 0, 'ALA': 0, 'VAL': 0, 'ILE': 0,
    'LEU': 0, 'MET': 0, 'PHE': 0, 'CYS': 1
}

# Size/volume
SIZE = {
    'GLY': 60, 'ALA': 89, 'SER': 93, 'CYS': 105, 'ASP': 111, 'PRO': 123,
    'ASN': 114, 'THR': 116, 'GLU': 138, 'VAL': 140, 'GLN': 144, 'HIS': 153,
    'MET': 163, 'ILE': 167, 'LEU': 167, 'LYS': 168, 'ARG': 173, 'PHE': 189,
    'TYR': 193, 'TRP': 228
}

# Simplified atom types for feature extraction
PROTEIN_ATOM_TYPES = {
    'C': 0, 'CA': 1, 'CB': 2, 'CD': 3, 'CD1': 4, 'CD2': 5, 'CE': 6, 'CE1': 7, 'CE2': 8,
    'CE3': 9, 'CG': 10, 'CG1': 11, 'CG2': 12, 'CH2': 13, 'CZ': 14, 'CZ2': 15, 'CZ3': 16,
    'N': 17, 'ND1': 18, 'ND2': 19, 'NE': 20, 'NE1': 21, 'NE2': 22, 'NH1': 23, 'NH2': 24,
    'NZ': 25, 'O': 26, 'OD1': 27, 'OD2': 28, 'OE1': 29, 'OE2': 30, 'OG': 31, 'OG1': 32,
    'OH': 33, 'OXT': 34, 'SD': 35, 'SG': 36, 'UNKNOWN': 37
}

LIGAND_ATOM_TYPES = {
    'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'F': 5, 'Cl': 6, 'Br': 7, 'I': 8, 'UNKNOWN': 9
}

class ProtLigNetParser:
    """
    A class for parsing and analyzing protein-ligand complexes for ML-based optimization.
    This class provides methods to:
    1. Parse PDB files and extract protein chains and ligands
    2. Identify binding site residues
    3. Generate modified structures for ML training
    4. Create graph representations of protein-ligand interactions
    5. Extract features for machine learning
    """

    def __init__(self, pdb_path: str, output_dir: str = "output", binding_cutoff: float = 5.0):
        """
        Initialize the parser with a PDB file.
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
        self.binding_graph = None
        self.features = {}
        # Load the PDB file
        self._load_pdb()
        logger.info(f"Initialized ProtLigNetParser with {pdb_path}")

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
            
            # Use MDAnalysis distance calculation for efficiency
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

    def generate_binding_site_files(self, remove_sidechains: bool = True):
        """
        Generate PDB files for each binding residue:
        1. One file with the ligand only
        2. One file with the protein and the binding residue modified (side chain stripped)
        Args:
            remove_sidechains: Whether to generate files with removed sidechains
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
            
            # Save original ligand
            ligand_path = ligand_dir / f"{lig_key}.pdb"
            ligand.write(str(ligand_path))
            
            # For each binding residue, create modified protein
            for res in binding_res:
                res_id = f"{res.resname}_{res.resid}"
                output_files[f"{lig_key}_{res_id}"] = {}
                
                # Save unmodified ligand for this residue
                output_files[f"{lig_key}_{res_id}"]["ligand"] = ligand_path
                
                if remove_sidechains:
                    # Create modified protein with the binding residue's side chain removed
                    # Keep only backbone atoms (N, CA, C, O) for the binding residue
                    backbone_atoms = ["N", "CA", "C", "O"]
                    
                    # Create a selection for all protein atoms except the side chain of the current residue
                    selection = f"(protein and not (resid {res.resid} and not name {' '.join(backbone_atoms)}))"
                    
                    # Create a new universe by slicing
                    modified_protein = self.universe.select_atoms(selection)
                    
                    # Save alpha carbon coordinates
                    ca_atom = self.universe.select_atoms(f"resid {res.resid} and name CA")
                    if len(ca_atom) > 0:
                        ca_coords = ca_atom.positions[0]
                    else:
                        logger.warning(f"No alpha carbon found for residue {res_id}")
                        ca_coords = None
                    
                    # Save modified protein
                    modified_path = ligand_dir / f"{lig_key}_{res_id}_modified.pdb"
                    modified_protein.write(str(modified_path))
                    output_files[f"{lig_key}_{res_id}"]["modified_protein"] = modified_path
                    output_files[f"{lig_key}_{res_id}"]["ca_coords"] = ca_coords
                    output_files[f"{lig_key}_{res_id}"]["original_resname"] = res.resname
                    
                    logger.info(f"Generated modified protein for binding residue {res_id}")
        
        # Save metadata to CSV
        metadata = []
        for key, files in output_files.items():
            if "ca_coords" in files and files["ca_coords"] is not None:
                lig_key, res_id = key.split('_', 1)
                row = {
                    "ligand": lig_key,
                    "residue": res_id,
                    "original_resname": files.get("original_resname", ""),
                    "ca_x": files["ca_coords"][0],
                    "ca_y": files["ca_coords"][1],
                    "ca_z": files["ca_coords"][2],
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

    def create_binding_graph(self, detailed_interactions: bool = True):
        """
        Create a graph representation of protein-ligand binding interactions.
        Args:
            detailed_interactions: Whether to compute detailed interaction types
        Returns:
            NetworkX graph representing the binding interactions
        """
        if not self.binding_residues:
            logger.error("Binding residues not identified. Run identify_binding_residues() first.")
            return None
        
        # Create a graph
        G = nx.Graph()
        
        for lig_key, binding_res in self.binding_residues.items():
            ligand_idx = int(lig_key.split('_')[-1]) - 1
            ligand = self.ligands[ligand_idx]
            
            # Add ligand node
            G.add_node(lig_key, type='ligand', atoms=len(ligand.atoms))
            
            # Add protein residue nodes and edges
            for res in binding_res:
                res_id = f"{res.resname}_{res.resid}"
                
                # Add protein residue node with properties
                G.add_node(res_id,
                          type='protein',
                          resname=res.resname,
                          resid=res.resid,
                          hydrophobicity=HYDROPHOBICITY.get(res.resname, 0),
                          charge=CHARGE.get(res.resname, 0),
                          h_bond=H_BOND.get(res.resname, 0),
                          size=SIZE.get(res.resname, 0))
                
                # Calculate minimum distance between ligand and residue
                ligand_coords = ligand.positions
                res_coords = res.atoms.positions
                distances_matrix = cdist(res_coords, ligand_coords)
                min_distance = np.min(distances_matrix)
                
                # Add edge with distance information
                G.add_edge(lig_key, res_id, distance=min_distance)
                
                if detailed_interactions and HAS_RDKIT:
                    # This is a placeholder for more detailed interaction analysis
                    # In a real implementation, you would analyze H-bonds, π-stacking, etc.
                    interaction_types = self._analyze_interaction_types(ligand, res)
                    for int_type, strength in interaction_types.items():
                        if strength > 0:
                            G[lig_key][res_id][int_type] = strength
        
        self.binding_graph = G
        logger.info(f"Created binding graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
        return G

    def _analyze_interaction_types(self, ligand, residue):
        """
        Analyze the types of interactions between a ligand and a residue.
        This is a placeholder for a more sophisticated analysis.
        Args:
            ligand: MDAnalysis AtomGroup for the ligand
            residue: MDAnalysis Residue object
        Returns:
            Dictionary of interaction types and strengths
        """
        interaction_types = {
            'h_bond': 0,
            'hydrophobic': 0,
            'ionic': 0,
            'pi_stacking': 0,
            'pi_cation': 0
        }
        
        # Simple heuristic: use residue properties to estimate interaction types
        resname = residue.resname
        
        # Hydrophobic interactions
        if HYDROPHOBICITY.get(resname, 0) > 0:
            interaction_types['hydrophobic'] = HYDROPHOBICITY.get(resname, 0) / 5.0  # Normalize
        
        # Hydrogen bonds
        if H_BOND.get(resname, 0) > 0:
            interaction_types['h_bond'] = H_BOND.get(resname, 0) / 6.0  # Normalize
        
        # Ionic interactions
        if abs(CHARGE.get(resname, 0)) > 0:
            interaction_types['ionic'] = abs(CHARGE.get(resname, 0))
        
        # Pi-stacking and pi-cation interactions
        if resname in ['PHE', 'TYR', 'TRP', 'HIS']:
            interaction_types['pi_stacking'] = 0.8
        
        if resname in ['ARG', 'LYS'] and CHARGE.get(resname, 0) > 0:
            interaction_types['pi_cation'] = 0.7
        
        return interaction_types

    def extract_features(self, radius: float = 12.0, max_neighbors: int = 32):
        """
        Extract features for machine learning, focusing on local environment.
        Args:
            radius: Radius around binding site to consider (in Å)
            max_neighbors: Maximum number of neighbors to consider
        Returns:
            Dictionary of features for each binding residue
        """
        if not self.binding_residues:
            logger.error("Binding residues not identified. Run identify_binding_residues() first.")
            return None
        
        features = {}
        for lig_key, binding_res in self.binding_residues.items():
            ligand_idx = int(lig_key.split('_')[-1]) - 1
            ligand = self.ligands[ligand_idx]
            
            for res in binding_res:
                res_id = f"{res.resname}_{res.resid}"
                features[f"{lig_key}_{res_id}"] = {}
                
                # 1. Residue features
                features[f"{lig_key}_{res_id}"]['residue'] = {
                    'resname': res.resname,
                    'resid': res.resid,
                    'hydrophobicity': HYDROPHOBICITY.get(res.resname, 0),
                    'charge': CHARGE.get(res.resname, 0),
                    'h_bond': H_BOND.get(res.resname, 0),
                    'size': SIZE.get(res.resname, 0)
                }
                
                # 2. Local environment features
                # Find all residues within radius of the current residue
                res_center = res.atoms.center_of_mass()
                
                # Select nearby residues
                nearby_residues = self.protein.select_atoms(f"point {res_center[0]} {res_center[1]} {res_center[2]} {radius}")
                
                # Get unique residues
                unique_nearby_residues = list(nearby_residues.residues)
                
                # Sort by distance to center
                residue_centers = np.array([r.atoms.center_of_mass() for r in unique_nearby_residues])
                distances_to_center = np.linalg.norm(residue_centers - res_center, axis=1)
                sorted_indices = np.argsort(distances_to_center)
                
                # Limit to max_neighbors
                if len(sorted_indices) > max_neighbors:
                    sorted_indices = sorted_indices[:max_neighbors]
                
                # Create neighbor features
                neighbor_features = []
                for idx in sorted_indices:
                    neighbor = unique_nearby_residues[idx]
                    # Skip the central residue
                    if neighbor.resid == res.resid:
                        continue
                    
                    neighbor_features.append({
                        'resname': neighbor.resname,
                        'resid': neighbor.resid,
                        'distance': distances_to_center[idx],
                        'hydrophobicity': HYDROPHOBICITY.get(neighbor.resname, 0),
                        'charge': CHARGE.get(neighbor.resname, 0),
                        'h_bond': H_BOND.get(neighbor.resname, 0),
                        'size': SIZE.get(neighbor.resname, 0)
                    })
                
                features[f"{lig_key}_{res_id}"]['neighbors'] = neighbor_features
                
                # 3. Ligand features (simplified)
                features[f"{lig_key}_{res_id}"]['ligand'] = {
                    'name': lig_key,
                    'num_atoms': len(ligand.atoms),
                    'num_heavy_atoms': len(ligand.select_atoms("not element H")),
                    'formal_charge': 0,  # Placeholder - would need RDKit for accurate charge
                    'molecular_weight': sum([atom.mass for atom in ligand.atoms])
                }
                
                # 4. Create atom-level features if RDKit is available
                if HAS_RDKIT:
                    try:
                        # This requires converting the ligand to SMILES and back to capture chemical properties
                        # In a real implementation, this would need to be expanded
                        atom_features = self._extract_atom_features(ligand)
                        features[f"{lig_key}_{res_id}"]['atom_features'] = atom_features
                    except Exception as e:
                        logger.warning(f"Failed to extract atom features: {e}")
                
                logger.info(f"Extracted features for binding residue {res_id}")
        
        self.features = features
        
        # Save features to JSON
        import json
        features_path = self.output_dir / "binding_features.json"
        
        # Convert complex objects (numpy arrays) to lists
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(i) for i in obj]
            else:
                return obj
        
        serialized_features = convert_to_json_serializable(features)
        with open(features_path, 'w') as f:
            json.dump(serialized_features, f, indent=2)
        
        logger.info(f"Saved binding features to {features_path}")
        return features

    def _extract_atom_features(self, ligand):
        """
        Extract atom-level features for the ligand.
        This is a placeholder for a more sophisticated analysis.
        Args:
            ligand: MDAnalysis AtomGroup for the ligand
        Returns:
            List of atom features
        """
        # In a real implementation, this would convert to RDKit and extract proper features
        atom_features = []
        for atom in ligand.atoms:
            # Get atom type (simplifying to element only)
            element = atom.element if hasattr(atom, 'element') else atom.name[0]
            
            # Basic features
            features = {
                'element': element,
                'position': atom.position.tolist(),
                'name': atom.name,
                'mass': atom.mass,
                'charge': atom.charge if hasattr(atom, 'charge') else 0.0
            }
            atom_features.append(features)
        
        return atom_features

    def visualize_binding_graph(self, output_path: Optional[str] = None):
        if self.binding_graph is None:
        logger.error("Binding graph not created. Run create_binding_graph() first.")
            return None
    
    G = self.binding_graph
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Get node types
    ligand_nodes = [node for node, attrs in G.nodes(data=True) if attrs.get('type') == 'ligand']
    protein_nodes = [node for node, attrs in G.nodes(data=True) if attrs.get('type') == 'protein']
    
    # Create layout for the graph
    pos = nx.spring_layout(G, seed=42)
    
    # Draw ligand nodes
    nx.draw_networkx_nodes(G, pos, nodelist=ligand_nodes, node_color='orange', 
                          node_size=800, label='Ligands')
    
    # Draw protein nodes with colors based on hydrophobicity
    hydrophobicity_values = [G.nodes[node].get('hydrophobicity', 0) for node in protein_nodes]
    # Normalize to range 0-1 for colormap
    if hydrophobicity_values:
        min_h = min(hydrophobicity_values)
        max_h = max(hydrophobicity_values)
        if min_h != max_h:
            normalized_values = [(h - min_h) / (max_h - min_h) for h in hydrophobicity_values]
        else:
            normalized_values = [0.5 for _ in hydrophobicity_values]
        nx.draw_networkx_nodes(G, pos, nodelist=protein_nodes, 
                              node_color=normalized_values, cmap=plt.cm.Blues,
                              node_size=500, label='Protein Residues')
    else:
        nx.draw_networkx_nodes(G, pos, nodelist=protein_nodes, 
                              node_color='blue', node_size=500, label='Protein Residues')
    
    # Draw edges with width based on distance (closer = thicker)
    edge_widths = [1/G[u][v].get('distance', 1) * 5 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7)
    
    # Draw labels
    ligand_labels = {node: node for node in ligand_nodes}
    protein_labels = {node: node.split('_')[0] for node in protein_nodes}  # Show only resname
    
    nx.draw_networkx_labels(G, pos, labels=ligand_labels, font_size=8, font_weight='bold')
    nx.draw_networkx_labels(G, pos, labels=protein_labels, font_size=8)
    
    plt.title("Protein-Ligand Binding Interactions")
    plt.axis('off')
    
    # Add a colorbar for hydrophobicity
    if protein_nodes and max(hydrophobicity_values) != min(hydrophobicity_values):
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, 
                                  norm=plt.Normalize(min_h, max_h))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label('Hydrophobicity')
    
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved binding graph visualization to {output_path}")
    
    return plt.gcf()


def main():
    """
    Main function to demonstrate the use of the ProtLigNetParser class.
    This function can be used as a command-line interface.
    """
    parser = argparse.ArgumentParser(description='Parse and analyze protein-ligand complexes.')
    parser.add_argument('pdb_file', type=str, help='Path to PDB file')
    parser.add_argument('--output', '-o', type=str, default='output', 
                        help='Directory to save output files')
    parser.add_argument('--binding-cutoff', '-b', type=float, default=5.0,
                        help='Distance cutoff (in Å) for defining binding site residues')
    parser.add_argument('--ligand-resnames', '-l', type=str, nargs='+',
                        help='Residue names to consider as ligands (optional)')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Visualize binding graph')
    
    args = parser.parse_args()
    
    # Initialize the parser
    protlignet = ProtLigNetParser(args.pdb_file, args.output, args.binding_cutoff)
    
    # Identify components
    protein, ligands = protlignet.identify_components(ligand_resnames=args.ligand_resnames)
    
    # Print summary
    print(f"Found protein with {len(protein.residues)} residues")
    print(f"Found {len(ligands)} ligand(s)")
    
    # Save components
    output_files = protlignet.save_components()
    print(f"Saved {len(output_files)} component files to {args.output}")
    
    # Identify binding residues
    if ligands:
        binding_residues = protlignet.identify_binding_residues()
        for lig_key, residues in binding_residues.items():
            print(f"Found {len(residues)} binding residues for ligand {lig_key}")
        
        # Create binding graph
        binding_graph = protlignet.create_binding_graph()
        
        # Visualize if requested
        if args.visualize:
            vis_path = Path(args.output) / "binding_graph.png"
            protlignet.visualize_binding_graph(str(vis_path))
    
    return protlignet


if __name__ == "__main__":
    main()
