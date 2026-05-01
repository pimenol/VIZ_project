from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class _ProteinBackbone:
  atom_positions: np.ndarray  # [num_res, 3, 3] for N, CA, C
  atom_mask: np.ndarray       # [num_res, 3]
  chain_index: np.ndarray     # [num_res]
  b_factors: np.ndarray       # [num_res, 3]


def _parse_pdb_backbone(pdb_str: str) -> '_ProteinBackbone':
  BACKBONE = {'N': 0, 'CA': 1, 'C': 2}
  residues: Dict = {}
  chain_order: Dict[str, int] = {}

  for line in pdb_str.splitlines():
    if not line.startswith('ATOM'):
      continue
    alt_loc = line[16]
    if alt_loc not in (' ', 'A'):
      continue
    atom_name = line[12:16].strip()
    if atom_name not in BACKBONE:
      continue
    chain_id = line[21]
    res_num = int(line[22:26])
    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
    b = float(line[60:66])

    if chain_id not in chain_order:
      chain_order[chain_id] = len(chain_order)
    key = (chain_id, res_num)
    if key not in residues:
      residues[key] = {}
    residues[key][atom_name] = (np.array([x, y, z]), b)

  sorted_keys = sorted(residues, key=lambda k: (chain_order[k[0]], k[1]))
  num_res = len(sorted_keys)
  atom_positions = np.zeros((num_res, 3, 3))
  atom_mask = np.zeros((num_res, 3))
  chain_index = np.zeros(num_res, dtype=int)
  b_factors = np.zeros((num_res, 3))

  for i, key in enumerate(sorted_keys):
    chain_index[i] = chain_order[key[0]]
    for name, idx in BACKBONE.items():
      if name in residues[key]:
        xyz, b = residues[key][name]
        atom_positions[i, idx] = xyz
        atom_mask[i, idx] = 1.0
        b_factors[i, idx] = b

  return _ProteinBackbone(atom_positions, atom_mask, chain_index, b_factors)


def _dihedral_angle(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> np.ndarray:
  b0 = p0 - p1
  b1 = p2 - p1
  b2 = p3 - p2
  n1 = np.cross(b0, b1)
  n2 = np.cross(b1, b2)
  b1_norm = b1 / (np.linalg.norm(b1, axis=-1, keepdims=True) + 1e-8)
  m1 = np.cross(n1, b1_norm)
  x = np.sum(n1 * n2, axis=-1)
  y = np.sum(m1 * n2, axis=-1)
  return np.degrees(np.arctan2(y, x))


def _compute_phi_psi(
    atom_positions: np.ndarray,
    atom_mask: np.ndarray,
    chain_index: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
  num_res = atom_positions.shape[0]
  n_pos = atom_positions[:, 0]
  ca_pos = atom_positions[:, 1]
  c_pos = atom_positions[:, 2]
  backbone_present = atom_mask[:, 0] * atom_mask[:, 1] * atom_mask[:, 2]

  phi = np.full(num_res, np.nan)
  psi = np.full(num_res, np.nan)

  # Phi[i] = dihedral(C[i-1], N[i], CA[i], C[i]) for i >= 1, same chain.
  same_chain_prev = chain_index[1:] == chain_index[:-1]
  valid_phi = same_chain_prev & (backbone_present[:-1] > 0) & (backbone_present[1:] > 0)
  if np.any(valid_phi):
    phi[1:][valid_phi] = _dihedral_angle(
        c_pos[:-1][valid_phi], n_pos[1:][valid_phi],
        ca_pos[1:][valid_phi], c_pos[1:][valid_phi])

  # Psi[i] = dihedral(N[i], CA[i], C[i], N[i+1]) for i < num_res-1, same chain.
  same_chain_next = chain_index[:-1] == chain_index[1:]
  valid_psi = same_chain_next & (backbone_present[:-1] > 0) & (backbone_present[1:] > 0)
  if np.any(valid_psi):
    psi[:-1][valid_psi] = _dihedral_angle(
        n_pos[:-1][valid_psi], ca_pos[:-1][valid_psi],
        c_pos[:-1][valid_psi], n_pos[1:][valid_psi])

  return phi, psi


def _classify_secondary_structure(
    phi: np.ndarray, psi: np.ndarray
) -> np.ndarray:
  ss = np.full(len(phi), 2, dtype=int)
  helix = (phi >= -160) & (phi <= -20) & (psi >= -80) & (psi <= 50)
  sheet = ((phi >= -180) & (phi <= -40) &
           (((psi >= 50) & (psi <= 180)) | ((psi >= -180) & (psi <= -90))))
  ss[helix] = 0
  ss[sheet] = 1
  return ss


def describe_protein_structure(pdb_path: str) -> np.ndarray:
  """Return per-residue SS labels (0=helix, 1=sheet, 2=loop) from a PDB file."""
  with open(pdb_path, 'r') as f:
    pdb_str = f.read()
  prot = _parse_pdb_backbone(pdb_str)
  phi, psi = _compute_phi_psi(prot.atom_positions, prot.atom_mask, prot.chain_index)
  return _classify_secondary_structure(phi, psi)
