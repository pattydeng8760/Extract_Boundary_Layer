from extract_boundary_layer_core import ProbeDataExporter, BoundaryLayerExtractor
import os
import argparse
from types import SimpleNamespace

def parse_args():
    parser = argparse.ArgumentParser(description="Boundary Layer and Probe Data Exporter")
    parser.add_argument("--alpha", type=float, default=10, help="Angle of attack in degrees")
    parser.add_argument("--uref", type=float, default=30.0, help="Reference freestream velocity in m/s")
    parser.add_argument("--rhoref", type=float, default=1.225, help="Reference density in kg/m^3")
    parser.add_argument("--pref", type=float, default=101325.0, help="Reference pressure in Pa")
    parser.add_argument("--mu_lam", type=float, default=1.78e-5, help="Reference Dynamic viscosity in kg/(m.s)")
    parser.add_argument("--chord", type=float, default=0.3048, help="Reference Chord length in meters")
    parser.add_argument("--input_dir", type=str, default='./' , help="Input directory path containing the mesh and solution files")
    parser.add_argument("--mesh", type=str, required=True, help="Mesh file location relative to the input directory")
    parser.add_argument("--solution", type=str, required=True, help="Averaged Solution file location relative to the input directory")
    parser.add_argument("--probe_files", type=str, nargs='+', required=True, 
                        help="List of .h5 probe files to process. Note: The files are extracted after time-series extraction should be in the same directory as the script.")
    parser.add_argument("--nb_pts", type=int, default=750, help="Number of points for the boundary layer extraction in the wall normal direction")
    parser.add_argument("--h_max", type=float, default=0.015, help="Maximum height for the boundary layer extraction in meters")
    parser.add_argument("--thresh", type=float, default=0.4, help="Threshold for the boundary layer extraction based on pressure coefficient")
    return parser.parse_args()

def extract_boundary_layer(args=None):
    """
    This is the main function that orchestrates the execution of the script to extracto the boundary layer parameters
    
    """
    print(f'\n{"Beginning Boundary Layer Extraction":=^100}\n')
    if isinstance(args, dict):
        args = SimpleNamespace(**args)
    elif args is None:
        args = parse_args()
        
    for probe_file in args.probe_files:
        try:
            probe_path = os.path.abspath(probe_file)
            print(f"\nProcessing: {probe_path}\n")

            exporter = ProbeDataExporter(
                probe_path, chord=args.chord, Uref=args.uref, alpha=args.alpha
            )
            exporter.export_coordinates()
            exporter.export_pressure_spectra()

            extractor = BoundaryLayerExtractor(
                args.input_dir,
                args.mesh,
                args.solution,
                probe_path,
                args.uref,
                args.rhoref,
                args.pref,
                args.mu_lam,
                args.nb_pts,
                args.h_max,
                args.thresh,
                chord=args.chord,
                alpha=args.alpha
            )
            extractor.run()
        except Exception as e:
            print(f"Failed to process {probe_file}: {e}")
        print(f'\n{"Complete Boundary Layer Extraction":=^100}\n')

if __name__ == "__main__":
    extract_boundary_layer()
