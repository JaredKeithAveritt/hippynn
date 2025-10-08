
import os
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # i think this can be removed, This is for when I had issues in Windows Subsystem for Linux 

import sys
import torch
# sys.path.append("../../programs/ANI-Tools-update/lib") # I think this can be removed, as its in the current version of hippynn

# Ani specific helpers
from hippynn.databases.h5_pyanitools import PyAniFileDB
AVAIL_METHODS, AVAIL_BASIS = ['hf', 'wb97x', 'ccsd(t)', 'mp2'], ['dz', 'tz', 'qz', 'cbs']
ANI1X_DSETS_KEYS = [
    'hf_tz.energy', 'coordinates', 'tpno_ccsd(t)_dz.corr_energy', 'wb97x_dz.hirshfeld_charges', 
    'wb97x_tz.mbis_charges', 'wb97x_tz.forces', 'mp2_tz.corr_energy', 'npno_ccsd(t)_tz.corr_energy', 
    'wb97x_tz.mbis_volumes', 'wb97x_tz.energy', 'wb97x_tz.dipole', 'wb97x_tz.mbis_octupoles', 
    'wb97x_tz.mbis_quadrupoles', 'mp2_qz.corr_energy', 'wb97x_tz.mbis_dipoles', 'wb97x_dz.cm5_charges', 
    'path', 'atomic_numbers', 'hf_qz.energy', 'mp2_dz.corr_energy', 'wb97x_dz.dipole', 
    'npno_ccsd(t)_dz.corr_energy', 'wb97x_dz.energy', 'hf_dz.energy', 'wb97x_dz.quadrupole', 
    'ccsd(t)_cbs.energy', 'wb97x_dz.forces'
]

# Helper Function for ani
def load_db(db_info, en_name, force_name, seed, location, n_workers):
    torch.set_default_dtype(torch.float64)
    return PyAniFileDB(
        file=location, species_key='species', seed=seed, num_workers=n_workers, 
        allow_unfound=True, 
        **db_info
    )
# Helper Function for ani
def get_data_names(qm_method, basis_set, force_training=False):
    assert qm_method in AVAIL_METHODS, f"Method not found: {qm_method}"
    assert basis_set in AVAIL_BASIS, f"Basis set not found: {basis_set}"
    spec = f"{qm_method}_{basis_set}"
    en_name = f"{spec}.energy"
    assert en_name in ANI1X_DSETS_KEYS, f"Data spec not available: {spec}"
    if force_training:
        assert f"{spec}.forces" in ANI1X_DSETS_KEYS, f"No force training for: {spec}"
    return en_name, f"{spec}.forces"

# ani specific
force_training = False
qm_method, basis_set = 'wb97x', 'dz'
en_name, force_name = get_data_names(qm_method, basis_set, force_training)




#Load .h5 style  database
from hippynn.databases.h5_pyanitools import PyAniFileDB

inputs = ['coordinates', 'species']
targets = ['energies', 'forces']

ani_base_database = load_db(db_info, 
                            en_name, 
                            force_name, 
                            #allow_unfound=True,
                            seed=101, 
                            location='ANI-2x-wb97xdz.h5', 
                            n_workers=2)

# Load .npz style database
from hippynn.databases import NPZDatabase

# Define the inputs and targets
inputs = ['coordinates', 'species']
targets = ['energy', 'forces']

# Initialize the base hippynn database
Zn_base_database=NPZDatabase(
                     file='Zn-all-AE.npz', 
                     seed=101, 
                     allow_unfound=True,
                     inputs=inputs,
                     targets=targets,
                     quiet=False
)

import sys

#can toggle between a h5 (ani) and npz (Al) database on lines 86, 87

from hippynn.databases.metadatabase import MetaDatabase
db_info = {"inputs": inputs , "targets": targets}

meta_database = MetaDatabase(
    arr_dict=Zn_base_database.arr_dict,
#    arr_dict=ani_base_database.arr_dict, 
    inputs=inputs,
    targets=targets,
    seed=12345,
    num_workers=1,
    pin_memory=True,
    allow_unfound=True,
    quiet=True,        
    species_key='species',
    coordinates_key='coordinates',
    energies_key='energy',
    forces_key='forces',
    metadata={ 
        "Energy_unit" : 'eV',
        "Mass_unit" : 'grams/mol', 
        "Distance_unit" : 'Angstroms',
        "Electronic_Structure_Package" : '',
        "Electronic_Structure_Package_Version" : '',
        "Computer_System" : '',
        "Input_Proceedure" : '' 
    },
    populate_metadata=True,
    write_metadata_to_json=True,
    json_filename='metadata.json',
    distribution_plots=False, # Need to fix bug, plotting will not work unless ran seperatly

)

# Need to fix bug, plotting will not work unless ran seperatly
# meta_database.plot_distributions()
