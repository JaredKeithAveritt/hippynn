# Need to add function to read_extxyz()

from pathlib import Path
import numpy as np
import torch  # needed by the function
from ase.data import atomic_numbers
from hippynn.databases import NPZDatabase

# Map element symbols -> atomic numbers if your NPZ uses strings
# numpy_map_elements = np.vectorize(atomic_numbers.__getitem__)

# Load your database
inputs = ['coordinates', 'species']
targets = ['energy', 'forces']
db = NPZDatabase(
    file='Zn-all-AE.npz',
    seed=101,
    allow_unfound=True,
    inputs=inputs,
    targets=targets,
    quiet=False
)

from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import write as ase_write

def write_extxyz(
    db, 
    file : str,
    record_split_masks: bool = True,
    overwrite : bool = False,
    split: Union[str, None] = None,
    Union[str, None] = None,
    pbc=None):
    """
    db.arr_dict must contain:
      coordinates: (n, max_atoms, 3)
      species:     (n, max_atoms)  int, padded with <=0
      forces:      (n, max_atoms, 3)
      atomenergies:(n, max_atoms, 1) or (n, max_atoms)
      energy:      (n,)
      cell:        (n, 3, 3)
      stress:      (n, 3, 3)
    """
    if split is True:
        database = database.write_npz("", record_split_masks=True, return_only=True)
    elif split in database.splits:
        database = database.splits[split]
        database = {k: v.to("cpu").numpy() for k, v in database.items()}
    elif split is None:
        database = database.arr_dict
    else:
        raise Exception(f"Unknown split variable supplied (must be True, None, or str): {split:s}")


    if file is not None:
        if Path(file).exists():
            if overwrite:
                print("Overwriting extxyz file:", file)
                Path(file).unlink()
            else:
                raise FileExistsError(f"file {file:s} exists.")
        print("Saving extxyz file:", file)

# I think lines 65 - 70 can be removed (JKA)
#        packer = DataPacker(file)
#    else:
#        packer = None  

# stoped here


    file = Path(file)
    if out_path.exists():
        if not overwrite:
            raise FileExistsError(f"Path exists: {out_path}")
        out_path.unlink()
    print("Saving EXTXYZ file:", out_path)

    # Pull numpy views from the in-memory arrays
    def to_np(x):
        import torch
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    A = {k: to_np(v) for k, v in db.arr_dict.items()}
    n_frames = A["species"].shape[0]

    for i in range(n_frames):
        sp = A["species"][i]                            # (max_atoms,)
        mask = sp > 0                                   # valid atoms
        Z   = sp[mask].astype(int)
        R   = A["coordinates"][i][mask].astype(float)   # (nat,3)

        atoms = Atoms(positions=R, numbers=Z)

        # cell and periodic flags
        if "cell" in A:
            atoms.set_cell(A["cell"][i], scale_atoms=False)
            if pbc is not None:
                atoms.set_pbc(tuple(bool(b) for b in pbc))

        # per-atom arrays
        if "forces" in A:
            atoms.new_array("forces", A["forces"][i][mask])
        if "atomenergies" in A:
            ae = A["atomenergies"][i]
            if ae.ndim == 3 and ae.shape[-1] == 1:
                ae = ae[..., 0]
            atoms.new_array("atomenergies", ae[mask])

        # frame scalars
        if "energy" in A:
            atoms.info["energy"] = float(A["energy"][i])
        if "stress" in A:
            atoms.info["stress"] = A["stress"][i].reshape(9)   # 9 components in header

        ase_write(str(out_path), atoms, format="extxyz", append=True)
