"""
Wrapper file to run the boundary layer extraction and probe data export
"""
from extract_boundary_layer import extract_boundary_layer  # or whatever your runner is called

args_dict = {
    "uref": 30.0,
    "rhoref": 1.225,
    "pref": 101325.0,
    "mu_lam": 1.78e-5,
    "chord": 0.3048,
    "alpha": 10,
    "input_dir": "../../",
    "mesh": "MESH_ZONE_Nov24/Bombardier_10AOA_Combine_Nov24.mesh.h5",
    "solution": "PostProc/Average_Field/Averaged_Solution_Reduced_Variables.h5",
    "probe_files": [
        "Group_A_Probe_Data.h5", "Group_B_Probe_Data.h5", "Group_C_Probe_Data.h5",
        "Group_D_Probe_Data.h5", "Group_E_Probe_Data.h5", "Group_F_Probe_Data.h5"
    ],
    "nb_pts": 750,
    "h_max": 0.015,
    "thresh": 0.4
}

extract_boundary_layer(args_dict)
