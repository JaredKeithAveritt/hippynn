import torch
torch.set_default_dtype(torch.float32)

from hippynn.experiment.serialization import load_checkpoint_from_cwd
from hippynn.tools import active_directory

from hippynn.interfaces.lammps_interface import MLIAPInterface


if __name__ == "__main__":
    # Load trained model
    try:
        with active_directory("./TEST_ALUMINUM_MODEL_MULTILAYER", create=False):
            bundle = load_checkpoint_from_cwd(map_location="cpu", restore_db=False)
    except FileNotFoundError:
        raise FileNotFoundError("Model not found, run lammps_example_Al.py first!")

    model = bundle["training_modules"].model
    energy_node = model.node_from_name("HEnergy")

    unified = MLIAPInterface(energy_node, ["Al"], model_device=torch.device("cuda"))
    unified.pickle("mliap_unified_hippynn_Al_multilayer.pkl")
