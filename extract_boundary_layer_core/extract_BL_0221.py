"""
This is the old version of the code. The new version is in the module extract_boundary_layer 
"""
import numpy as np
import os
import csv
from antares import *
import h5py
from scipy.signal import savgol_filter
from scipy.signal import welch
import scipy.spatial
import scipy.integrate
import re


class ProbeDataExporter:
    """
    A class to export probe data from an HDF5 file into two types of CSV files
    (space-delimited):
    
    1. T10_BL_A_Coordinates.csv:
       This file contains one row per probe with 9 columns:
         idxBL xBL yBL zBL idxWPS xWPS yWPS zWPS dist
       Here:
         - idxBL is the zero-padded index (e.g., "000")
         - xBL, yBL, zBL are the BL coordinates read from each probe’s attributes.
         - idxWPS and xWPS, yWPS, zWPS are the same as BL values.
         - dist is set to 0.
    
    2. For each probe, a pressure spectrum file:
       For each probe group, the pressure signal (assumed stored in a dataset named "pressure")
       and the sampling time step dt (stored in the group attributes as "dt") are used to compute an FFT.
       The FFT is performed with n_fft = 510 so that np.fft.rfft produces exactly 256 bins.
       The frequency vector is normalized to a Strouhal number using:
           St = f * c / Uref
       The PSD (in dB/Hz) is computed as:
           PSD_dB = 10 * log10( PSD/(p_ref**2) )
       and saved in a file named "T10_WPS_A<sensor_tag_without_space>_Spectrum.csv".
    """
    
    def __init__(self, probe_filepath, chord:float=0.3048, Uref:float=30.0, alpha:int=10, LE_dist:float=1.245):
        """
        Parameters:
            probe_filepath: Path to the HDF5 file containing the probe data.
            chord: The chord length (default 0.3048 m).
            Uref: The reference (free-stream) velocity (default 30 m/s).
        """
        self.probe_filepath = probe_filepath
        self.chord = chord
        self.Uref = Uref
        self.alpha = alpha
        self.LE_dist = LE_dist
        match = re.search(r'Group_([A-Z])_Probe', probe_filepath)  # Searches for 'Group_A_Probe' pattern.
        if match:
            self.group_letter = match.group(1)
        new_dir = os.path.join(os.getcwd(),'T'+str(alpha)+'_Group_'+self.group_letter)
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        self.export_dir = new_dir
        # Will be populated by reading the file.
        self.probe_groups = []  # List of group names (e.g., "Probe_01", "Probe_02", ...)
    
    def _detect_probes(self):
        """Detects all groups starting with 'Probe_' in the HDF5 file."""
        with h5py.File(self.probe_filepath, "r") as f:
            self.probe_groups = sorted([key for key in f.keys() if key.startswith("Probe_")])
    
    def export_coordinates(self):
        """
        Reads each probe group's attributes (x, y, z) and writes a space-delimited CSV file
        named "T10_BL_A_Coordinates.csv" with the following 9 columns:
            idxBL xBL yBL zBL idxWPS xWPS yWPS zWPS dist
        The indices are zero-padded (e.g., "000") and the distance is set to 0.
        """
        self._detect_probes()
        rows = []
        # Loop over detected probe groups
        with h5py.File(self.probe_filepath, "r") as f:
            for i, group_name in enumerate(self.probe_groups):
                group = f[group_name]
                # Read coordinates from the group's attributes
                xBL = group.attrs["x"]
                yBL = group.attrs["y"]
                zBL = group.attrs["z"]
                xBL = ((xBL - self.LE_dist)*np.cos(self.alpha*np.pi/180))/self.chord
                yBL = yBL/self.chord
                # For this example, we assume the WPS data are the same as the BL data.
                idx_str = f"{i:03d}"
                row = [idx_str, xBL, yBL, zBL, idx_str, xBL, yBL, zBL, 0]
                rows.append(row)
        fmt = "{:03d} {:10.8f} {:10.8f} {:10.8f} {:03d} {:10.8f} {:10.8f} {:10.8f} {:2.4f} "  # Adjust width as needed
        filename = "T"+self.group_letter+f"{self.alpha:02d}_coordinates.csv"
        csv_filename = os.path.join(self.export_dir, filename)
        if os.path.exists(csv_filename):
            os.remove(csv_filename)
        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f, delimiter=" ")
            header = ["idxBL", "xBL", "yBL", "zBL", "idxWPS", "xWPS", "yWPS", "zWPS", "dist"]
            writer.writerow(header)
            for row in rows:
                f.write(fmt.format(int(row[0]), row[1], row[2], row[3], int(row[4]), row[5], row[6], row[7], row[8])+ "\n")  # Fixed-width formatting
        print("CSV coordinate file written:",  os.path.basename(os.path.normpath(csv_filename))) 
    
    def export_pressure_spectra(self):
        """
        For each probe group, extracts the pressure signal and dt (from attributes),
        computes a FFT with n_fft = 510 (which produces 256 bins via np.fft.rfft),
        normalizes the frequency to Strouhal number (f * chord / Uref), computes the PSD in dB/Hz,
        and writes the result to a space-delimited CSV file.
        
        Each file is named "T10A_WPS_<sensorTag>_Spectrum.csv" where sensorTag is the number
        extracted from the probe group (e.g. "A01", "A02", etc.).
        """
        self._detect_probes()
        with h5py.File(self.probe_filepath, "r") as f:
            for group_name in self.probe_groups:
                group = f[group_name]
                # Assume the pressure signal is stored in a dataset named "pressure"
                pressure_signal = group["Pressure"][:]
                dt = group.attrs["dt"]
                # Welch parameters
                nfft = 2048
                window = 'hann'  # or you can create a window via scipy.signal.get_window
                noverlap = 1024
                # Compute the PSD via Welch's method (Pa^2 / Hz)
                freq, PSD = welch(pressure_signal, fs=1/dt, window=window, nperseg=nfft, noverlap=noverlap, nfft=nfft, scaling='density')
                St = freq * self.chord / self.Uref
                # Convert to decibels relative to 20 micropascals
                p_ref = 2e-5  # Pa
                PSD_dB = 10 * np.log10(PSD / (p_ref**2))
                # Prepare the CSV data: two columns [St, psd]
                spectrum_data = np.column_stack((St, PSD_dB))
                # Extract sensor tag from group name. For example, "Probe_01" -> "A01"
                tag_num = group_name.split("_")[1]
                tag_num = str(int(tag_num)-1)
                f_ind = tag_num.zfill(3)
                filetemp  = "T"+self.group_letter+f"{self.alpha:02d}_WPS_"+f_ind+".csv"
                csv_filename = os.path.join(self.export_dir, filetemp)
                if os.path.exists(csv_filename):
                    os.remove(csv_filename)
                # Define a fixed-width format for each column (e.g., 15 characters wide, 8 decimal places)
                fmt = "{:12.10f} {:12.10f}"  # Adjust width as needed
                with open(csv_filename, "w", newline="") as f_csv:
                    writer = csv.writer(f_csv, delimiter=" ")  # We still use space as the separator, but...
                    # Write the header
                    header = ["strouhal","PSD[dB/Hz]"]
                    writer.writerow(header)
                    # Write data rows
                    for row in spectrum_data:
                        f_csv.write(fmt.format(row[0], row[1]) + "\n")  # Fixed-width formatting
                print("CSV WPS file written:",  os.path.basename(os.path.normpath(csv_filename))) 

class BoundaryLayerExtractor:
    """
    A class to load AVBP data, extract boundary layer parameters, and
    export an aggregated CSV file containing:
      id, zone, x, chord, Uref, Ue, delta, delta_star, theta, tau_w, cp, cf,
      beta_c, Rt, Rtheta, PI, u_tau, mu, rho, nu, dpdx

    The extraction procedure is divided into several steps:
      1. Read the mesh and mean solution and compute derived fields.
      2. Extract the airfoil profile (via a planar cut and line unwrapping).
      3. Separate the profile into suction and pressure sides.
      4. Compute local tangents, curvilinear abscissa, and dp/dS.
      5. At selected x‑locations, extract a BL line (using a normal direction) and compute:
           – BL thickness (via a zero‑crossing criterion on dP/dh)
           – Edge velocity, displacement thickness, momentum thickness,
           – Wall shear stress, pressure gradient, Cp and Cf.
      6. Compute additional nondimensional parameters and export all data to a CSV file.

    By default, both suction and pressure sides are processed unless the
    extraction_side flag is set to "suction" or "pressure".
    """
    
    def __init__(self, input_directory, avbp_mesh, avbp_mean_solution, probe_file, casename,
                uref, rhoref, pref, mu_lam, nb_pts, h_max,threshold_hmax_factor, 
                chord:float = 0.3048, alpha:int = 10, cut_z:float = -0.2694, LE_dist:float = 1.245):
        # Simulation and extraction parameters
        self.input_directory = input_directory
        self.avbp_mesh = avbp_mesh
        self.avbp_mean_solution = avbp_mean_solution
        self.casename = casename
        self.uref = uref
        self.rhoref = rhoref
        self.pref = pref
        self.mu_lam = mu_lam
        self.nb_pts = nb_pts
        self.h_max = h_max
        self.threshold_hmax_factor = threshold_hmax_factor
        self.chord = chord
        self.alpha = alpha
        self.cut_z = cut_z
        self.LE_dist = LE_dist
        #Loading the .h5 file for probe settings
        # Load probe data from from the probe file in .h5 format
        sensor_tags = []
        extract_points_xcoor = []
        with h5py.File(probe_file, "r") as f:
            probe_groups = sorted([key for key in f.keys() if key.startswith("Probe_")])
            for group in probe_groups:
                tag_num = group.split("_")[1]
                match = re.search(r'Group_([A-Z])_Probe', probe_file)  # Searches for 'Group_A_Probe' pattern.
                cut_z = f[group].attrs["z"]
                if match:
                    self.group_letter = match.group(1)
                sensor_tag = f"{self.group_letter}{tag_num}"
                sensor_tags.append(sensor_tag)
                # Read the x value from the group's attributes
                x_val = f[group].attrs["x"]
                extract_points_xcoor.append(x_val)
            if "Group_F" in probe_file:
                group1 = np.array([f['Probe_01'].attrs['x'], f['Probe_01'].attrs['y'], f['Probe_01'].attrs['z']])
                group9 = np.array([f['Probe_09'].attrs['x'], f['Probe_09'].attrs['y'], f['Probe_09'].attrs['z']])
                line_group = group9-group1
                #self.normal = np.array([-line_group[1], line_group[0], 0])
                self.normal = np.array([0,1,0])
                self.origin = group1
                self.extraction_side = 'tip'
            if "Group_A" or "Group_C" or "Group_D" in probe_file:
                self.extraction_side = 'suction'
            elif "Group_B" or "Group_E" in probe_file:
                self.extraction_side = 'pressure'
            else:
                self.normal = np.array([0, 0, 1])
                self.origin = np.array([0, 0, cut_z])
                self.extraction_side = 'both'
        
        self.extract_points_xcoor = extract_points_xcoor
        self.sensors_tag = sensor_tags
        new_dir = os.path.join(os.getcwd(),"T"+f"{self.alpha:02d}_Group_"+self.sensors_tag[0][0])
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        self.export_dir = new_dir
        # New flag: extraction_side can be "both" (default), "suction", or "pressure"
        #self.extraction_side = extraction_side.lower()
        # Data containers
        self.solution = None
        self.profile = None
        self.separated = None
        

    def load_data(self):
        """
        Reads the AVBP mesh and mean solution, and computes derived fields
        (total pressure, velocity magnitude, and Cp).
        """
        store_vtk(True)
        # Read mesh
        print('\n----> Reading the mesh')
        r = Reader('hdf_avbp')
        r['filename'] = os.path.join(self.input_directory, self.avbp_mesh)
        r['shared'] = True
        b = r.read()
        print('The mesh has been read')
        
        # Read mean solution
        print('\n----> Reading the mean solution')
        r = Reader('hdf_antares')
        r['filename'] = os.path.join(self.input_directory, self.avbp_mean_solution)
        r['base'] = b
        b = r.read()
        b.show()
        print('The mean solution has been read')
        
        # Compute derived quantities
        b.compute('Pt=P+0.5*rho*(u**2+v**2+w**2)')
        b.compute('Umag=(u**2+v**2+w**2)**0.5')
        b[0][0]['Cp'] = b[0][0]['P'] / (0.5 * self.rhoref * self.uref**2)
        
        self.solution = b

    def extract_profile(self):
        """
        Extracts the airfoil profile from the solution using a surface patch,
        a planar cut, and then unwraps the resulting line.
        """
        patches = self.solution[self.solution.families['Patches']]
        profil = patches['Airfoil_Surface', 'Airfoil_Side_LE',
                         'Airfoil_Side_Mid', 'Airfoil_Side_TE']
        
        # Merge all the zones to one unified surface
        myt = Treatment('merge')
        myt['base'] = profil
        myt['duplicates_detection'] = False
        myt['tolerance_decimals'] = 13
        merged = myt.execute()

        # Cut the surface with a plane to extract the profile at the specific line
        t = Treatment('cut')
        t['base'] = merged
        t['type'] = 'plane'
        t['origin'] = [0., 0., self.cut_z] if self.origin is None else self.origin        # Default cut_z value if not provided
        t['normal'] = [0., 0., 1.] if self.normal is None else self.normal                # Default normal value if not provided
        
        profil_2d = t.execute()
        # Unwrap the resulting line
        t = Treatment('unwrapline')
        t['base'] = profil_2d
        self.profile = t.execute()

    def separate_profile(self):
        """
        Splits the unwrapped airfoil profile into two zones (suction and pressure sides)
        using a leading‑ and trailing‑edge identification.
        """
        x_coord = self.profile[0][0]['x']
        y_coord = self.profile[0][0]['y']
        z_coord = self.profile[0][0]['z']
        coords = np.array([x_coord, y_coord, z_coord]).T
        
        # Determine leading and trailing edges via maximum distance between points.
        dist_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coords))
        idx_LE, idx_TE = np.unravel_index(np.argmax(dist_matrix), (coords.shape[0], coords.shape[0]))
        
        # Ensure idx_LE corresponds to the upstream (lower x) point
        if x_coord[idx_LE] > x_coord[idx_TE]:
            idx_LE, idx_TE = idx_TE, idx_LE
        
        LE_coords = coords[idx_LE, :]
        TE_coords = coords[idx_TE, :]
        
        vec_dir_chord = TE_coords - LE_coords
        chord = np.linalg.norm(vec_dir_chord)
        vec_dir_chord = vec_dir_chord / chord
        
        pz = np.array([0., 0., 1.0])
        chi_angle_sign = 1.0
        dist_from_LE = np.linalg.norm(coords - coords[idx_LE, :], axis=1)
        
        separated = Base()
        separated.init(zones=('suction_side', 'pressure_side'))
        
        # Split the profile based on the order of points
        if idx_TE > idx_LE:
            d2, indices_S1 = np.unique(dist_from_LE[idx_LE:idx_TE + 1], return_index=True)
            part1 = coords[idx_LE:idx_TE + 1, :][indices_S1]
            pseg1 = part1[1] - part1[0]
            vec_dir1 = np.cross(vec_dir_chord, pseg1)
            eval_sign = np.dot(vec_dir1, pz) * chi_angle_sign
            if eval_sign > 0:
                side1_name = 'suction_side'
                side2_name = 'pressure_side'
            else:
                side1_name = 'pressure_side'
                side2_name = 'suction_side'
            d2, indices_S2 = np.unique(np.concatenate([dist_from_LE[idx_TE:], dist_from_LE[:idx_LE + 1]]),
                                       return_index=True)
            for var in self.profile[0][0].keys():
                separated[side1_name][0][var] = self.profile[0][0][var][idx_LE:idx_TE + 1][indices_S1]
                separated[side2_name][0][var] = np.concatenate([self.profile[0][0][var][idx_TE:],
                                                                 self.profile[0][0][var][:idx_LE + 1]])[indices_S2]
        else:
            d2, indices_S1 = np.unique(np.concatenate([dist_from_LE[idx_LE:], dist_from_LE[:idx_TE + 1]]),
                                       return_index=True)
            part1 = np.concatenate([coords[idx_LE:, :], coords[:idx_TE + 1, :]])[indices_S1]
            pseg1 = part1[1] - part1[0]
            vec_dir1 = np.cross(vec_dir_chord, pseg1)
            eval_sign = np.dot(vec_dir1, pz) * chi_angle_sign
            if eval_sign > 0:
                side1_name = 'suction_side'
                side2_name = 'pressure_side'
            else:
                side1_name = 'pressure_side'
                side2_name = 'suction_side'
            d2, indices_S2 = np.unique(dist_from_LE[idx_TE:idx_LE + 1], return_index=True)
            for var in self.profile[0][0].keys():
                separated[side1_name][0][var] = np.concatenate([self.profile[0][0][var][idx_LE:],
                                                                 self.profile[0][0][var][:idx_TE + 1]])[indices_S1]
                separated[side2_name][0][var] = self.profile[0][0][var][idx_TE:idx_LE + 1][indices_S2]
        
        self.separated = separated

    def compute_profile_derivatives(self):
        """
        For each separated zone, compute the local tangents (first derivatives) and
        the curvilinear abscissa along the profile, and derive the pressure gradient dPdS.
        """
        for zn in self.separated.keys():
            # Compute tangents for each coordinate
            for coor in ['x', 'y', 'z']:
                data = self.separated[zn][0][coor]
                dx_c = data[1:] - data[:-1]
                dx = np.zeros_like(data)
                dx[1:-1] = (dx_c[1:] + dx_c[:-1]) * 0.5
                dx[0] = dx_c[0]
                dx[-1] = dx_c[-1]
                self.separated[zn][0]['d' + coor] = dx
            
            # Create curvilinear abscissa
            npts = np.size(self.separated[zn][0]['x'])
            ds = np.zeros(npts)
            ds[0] = 0
            ds[1:] = np.sqrt(
                (self.separated[zn][0]['x'][1:] - self.separated[zn][0]['x'][:-1])**2 +
                (self.separated[zn][0]['y'][1:] - self.separated[zn][0]['y'][:-1])**2 +
                (self.separated[zn][0]['z'][1:] - self.separated[zn][0]['z'][:-1])**2
            )
            s = np.cumsum(ds)
            s = s / s[-1]
            self.separated[zn][0]['abscisse_curviligne'] = s
            
            # Compute dP/dS along the profile
            ds_grad = np.gradient(s)
            dPsds = np.gradient(self.separated[zn][0]['P']) / ds_grad
            self.separated[zn][0]['dPdS'] = dPsds

    def extract_BL_parameters_and_export(self):
        """
        For each separated zone (or for the specified extraction_side) and for each specified extraction x-location,
        extract a BL line (using a line treatment normal to the profile), compute the boundary layer parameters,
        and also compute additional nondimensional parameters.
        
        The following additional parameters are computed:
          - Normalized x: (x - 1.225) / 0.3048
          - beta_c = (dp/dx * delta_star) / tau_w
          - Rt = Ue * delta / nu, where nu = mu / rho
          - Rtheta = Ue * theta / nu
          - PI = (dp/dx * theta) / tau_w
          - u_tau = sqrt(tau_w / rho)
          - Zone classification based on dp/dx:
              FPG if dp/dx < 0, APG if dp/dx > 0, and ZPG otherwise.
        
        The results for each extraction point (from the processed zone(s)) are written to a CSV file:
          T_10AoA_30_GroupA_Zones.csv
        
        Returns:
            A list of rows (as dictionaries) with the computed parameters.
        """
        csv_rows = []
        tol = 1e-5  # Tolerance for pressure gradient classification
        row_id = 0

        # Determine which zones to process based on the flag.
        if self.extraction_side in ["suction", "pressure"]:
            target_zone = self.extraction_side + "_side"
            zones_to_process = [target_zone] if target_zone in self.separated else []
        if self.extraction_side in ['tip']:
            zones_to_process = ['suction_side']
        else:
            zones_to_process = list(self.separated.keys())
        
        # Loop over each zone to process.
        for zn in zones_to_process:
            # For computing dp/dx from the separated profile:
            P_array = self.separated[zn][0]['P']
            x_array = self.separated[zn][0]['x']
            P_smooth = savgol_filter(P_array, window_length=121, polyorder=2)  # Smoothing the pressure field before computing the gradient
            dpdx_array = np.gradient(P_smooth, x_array)
            dpdx_array = savgol_filter(dpdx_array, window_length=151, polyorder=2)  # Smoothing the gradient
            export_filename = os.path.join(self.export_dir,'CFD_extraction_{0:s}_{1:s}.txt'.format(self.casename, zn))
            BL_data = np.zeros((len(self.extract_points_xcoor), 10))
            
            for ind, x_point in enumerate(self.extract_points_xcoor):
                # Find the closest point on the profile in the x direction.
                zone_obj, ind_extract, _ = self.separated[(zn,)].closest([x_point, ], coordinates=['x', ])
                Pt_extract = [self.separated[zn][0]['x'][ind_extract],
                              self.separated[zn][0]['y'][ind_extract],
                              self.separated[zn][0]['z'][ind_extract]]
                print("Extracting Boundary Layer at point index: {0}, coordinates: {1} of {2}".format(ind, np.array(Pt_extract),x_point))
                
                gradPds = self.separated[zn][0]['dPdS'][ind_extract]
                Cp = self.separated[zn][0]['Cp'][ind_extract]
                
                # Tangent vector at the extraction point
                T_extract = [self.separated[zn][0]['dx'][ind_extract],
                             self.separated[zn][0]['dy'][ind_extract],
                             self.separated[zn][0]['dz'][ind_extract]]
                norm_T_vec = np.linalg.norm(T_extract)
                T_extract = np.array(T_extract) / norm_T_vec
                
                # Corrected: index the wall stresses to get scalar values for Cf
                Cf = (self.separated[zn][0]['wall_Stress_x'][ind_extract] * T_extract[0] +
                      self.separated[zn][0]['wall_Stress_y'][ind_extract] * T_extract[1] +
                      self.separated[zn][0]['wall_Stress_z'][ind_extract] * T_extract[2])
                
                chi_angle_sign = 1.0
                pz = np.array([0., 0., 1.0])
                if zn == 'suction_side' and self.extraction_side == 'suction':
                    N_extract = np.cross(chi_angle_sign * pz, T_extract)
                elif zn == 'pressure_side' and self.extraction_side == 'pressure':
                    N_extract = np.cross(T_extract, chi_angle_sign * pz)
                elif self.extraction_side == 'tip':
                    N_extract = np.array([0,0,1])
                # Define the end point for the BL line (normal direction)
                Pt_extract_end_line = np.array(Pt_extract) + self.h_max * N_extract
                
                t_line = Treatment('line')
                t_line['base'] = self.solution
                t_line['point1'] = Pt_extract
                t_line['point2'] = Pt_extract_end_line.tolist()
                t_line['nbpoints'] = self.nb_pts
                line = t_line.execute()
                
                # Record the profile point for reference
                line.attrs['x_profil'] = Pt_extract[0]
                line.attrs['y_profil'] = Pt_extract[1]
                line.attrs['z_profil'] = Pt_extract[2]
                h_vals = np.sqrt((line[0][0]['x'] - Pt_extract[0])**2 +
                                 (line[0][0]['y'] - Pt_extract[1])**2 +
                                 (line[0][0]['z'] - Pt_extract[2])**2)
                line[0][0]['h'] = h_vals
                
                # Optionally export the BL line (here only if suction side is processed)
                # if zn == 'suction_side':
                #     w = Writer('column')
                #     outputname = os.path.join(self.export_dir, 'BL_extract_{0}_{1}'.format(self.sensors_tag[ind], zn))
                #     print("Exporting:", outputname)
                #     print("Normal vector:", N_extract)
                #     w['filename'] = outputname
                #     w['base'] = line
                #     w.dump()

                # Create a new h array: 150 evenly spaced points from 0 to 0.1
                h_new = np.linspace(0, 0.1, 150)
                # Original data arrays
                h_orig = line[0][0]['h']/self.chord  # original heights
                U_orig = line[0][0]['Umag']/self.uref  # original tangential speed
                # Interpolate tangential speed onto the new h values np.interp assumes that h_orig is in ascending order.
                U_new = np.interp(h_new, h_orig, U_orig)
                # Normalize the tangential speed by the exterior stream velocity Ue
                U_new_norm = U_new
                # Prepare and write the CSV file with two columns: h/chord and normalized tangential speed.
                csv_header = ["h", "Ut"]
                fmt = "{:10.8f} {:10.8f}"  # Adjust width as needed
                f_ind = str(ind).zfill(3)
                filetemp  = "T"+self.sensors_tag[ind][0]+f"{self.alpha:02d}_BL_"+f_ind+".csv"
                csv_filename = os.path.join(self.export_dir,filetemp)
                if os.path.exists(csv_filename):
                    os.remove(csv_filename)
                with open(csv_filename, "w", newline="") as f:
                    writer = csv.writer(f,delimiter=' ')
                    writer.writerow(csv_header)
                    for i in range(len(h_new)):
                        f.write(fmt.format(h_new[i], U_new_norm[i]) + "\n")  # Fixed-width formatting
                print("CSV file written:", csv_filename)
                
                # Calculating the pressure gradient
                dh = np.gradient(line[0][0]['h'])
                dPtdh = np.gradient(line[0][0]['Pt'], dh, edge_order=2)
                
                # Find the index corresponding to the zero crossing (delta99 criterion)
                zero_crossings = self.find_zero_crossing(dPtdh)
                found_zero = False
                if zero_crossings.size == 0:
                    idx_delta99 = 0
                elif zero_crossings.size >= 2:
                    idx_z0 = -1
                    check_cons = True
                    while check_cons:
                        idx_delta99 = zero_crossings[idx_z0]
                        if line[0][0]['h'][idx_delta99] > self.threshold_hmax_factor * self.h_max:
                            idx_z0 -= 1
                            if idx_z0 + zero_crossings.size < 0:
                                check_cons = False
                                idx_delta99 = 0
                        else:
                            check_cons = False
                            found_zero = True
                else:
                    idx_delta99 = zero_crossings[0]
                    found_zero = True
                
                if not found_zero:
                    idx_delta99 = np.argmax(line[0][0]['Pt'])
                    print("Criterium on total pressure not reached for cut {0}".format(ind))
                
                Ue = line[0][0]['Umag'][idx_delta99]
                # Compute displacement thickness (delta_star) and momentum thickness (theta)
                delta_star = scipy.integrate.simpson(
                    1.0 - line[0][0]['Umag'][:idx_delta99] / Ue,
                    x=line[0][0]['h'][:idx_delta99]
                )
                theta = scipy.integrate.simpson(
                    (1.0 - line[0][0]['Umag'][:idx_delta99] / Ue) * line[0][0]['Umag'][:idx_delta99] / Ue,
                    x=line[0][0]['h'][:idx_delta99]
                )
                dUdh = np.diff(line[0][0]['Umag'])/dh[0:-1]
                tau_w = self.mu_lam * dUdh[0]
                #dpdx = dpdx_array[ind_extract]
                dpdx = gradPds[ind_extract]
                
                # Local density extracted from the profile (assumed available)
                rho_local = self.separated[zn][0]['rho'][ind_extract]
                nu = self.mu_lam / rho_local
                
                beta_c = (dpdx * delta_star) / tau_w if tau_w != 0 else 0.0
                Rt = (Ue * line[0][0]['h'][idx_delta99] / nu)  if nu != 0 else 0.0
                Rtheta = (Ue * theta / nu) if nu != 0 else 0.0
                PI = (dpdx * theta) / tau_w if tau_w != 0 else 0.0
                u_tau = np.sqrt(tau_w / rho_local) if rho_local != 0 else 0.0
                
                if dpdx < -tol:
                    zone_class = "FPG"
                elif dpdx > tol:
                    zone_class = "APG"
                else:
                    zone_class = "ZPG"
                chord = self.chord
                x_norm = ((Pt_extract[0] - self.LE_dist)*np.cos(self.alpha*np.pi/180)) / chord
                print("extracting at {0} with original {1}".format(x_norm, Pt_extract[0]))
                y_norm = Pt_extract[1] / chord
                z_norm = np.abs(Pt_extract[2] / chord)
                csv_row = [
                    f"{row_id:03d}",
                    zone_class,
                    x_norm,
                    y_norm,
                    z_norm,
                    chord,
                    self.uref,
                    Ue,
                    line[0][0]['h'][idx_delta99],
                    delta_star,
                    theta,
                    tau_w,
                    Cp,
                    Cf,
                    beta_c,
                    Rt,
                    Rtheta,
                    PI,
                    u_tau,
                    self.mu_lam,
                    rho_local,
                    nu,
                    dpdx,
                ]
                csv_rows.append(csv_row)
                row_id += 1
                
                BL_data[ind, 0] = Pt_extract[0]
                BL_data[ind, 1] = Pt_extract[1]
                BL_data[ind, 2] = line[0][0]['h'][idx_delta99]
                BL_data[ind, 3] = Ue
                BL_data[ind, 4] = delta_star
                BL_data[ind, 5] = theta
                BL_data[ind, 6] = tau_w
                BL_data[ind, 7] = gradPds
                BL_data[ind, 8] = Cp
                BL_data[ind, 9] = Cf
            
            np.savetxt(export_filename, BL_data,
                       header='"x" "y" "bl_thickness" "exterior_stream_velocity" "displacement_thickness" "momentum_thickness" "tau_wall" "pressure_gradient" "Cp" "Cf"',
                       comments='')
        
        csv_header = ["id", "zone", "x", "y", "z", "chord", "Uref", "Ue", "delta", "delta_star", "theta", "tau_w",
                      "cp", "cf", "beta_c", "Rt", "Rtheta", "PI", "u_tau", "mu", "rho", "nu", "dpdx"]
        file_temp = "T"+self.sensors_tag[0][0]+f"{self.alpha:02d}_BLparams_Zones.csv"
        csv_filename = os.path.join(self.export_dir, file_temp)
        if os.path.exists(csv_filename):
            os.remove(csv_filename)
        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f,delimiter=' ')
            writer.writerow(csv_header)
            for row in csv_rows:
                writer.writerow(row)
        print("CSV file written:", csv_filename)
        return csv_rows
    
    def write_flow_conditions(self):
        """writing the flow conditions of the simulation to a csv
        """
        csv_header = ["Mach", "Reynolds", "alpha", "Uinf"]
        file_temp = "T"+self.sensors_tag[0][0]+f"{self.alpha:02d}_flowconditions.csv"
        csv_filename = os.path.join(self.export_dir, file_temp)
        mach = self.uref / 343.0
        reynolds = self.rhoref * self.uref * self.chord / self.mu_lam
        if os.path.exists(csv_filename):
            os.remove(csv_filename)
        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f,delimiter=' ')
            writer.writerow(csv_header)
            writer.writerow([mach, reynolds, self.alpha, self.uref])


    @staticmethod
    def find_zero_crossing(data):
        """
        Returns the indices where a zero crossing occurs in the data.
        """
        return np.where(np.diff(np.signbit(data)))[0]

    @staticmethod
    def find_nearest(array, value):
        """
        Returns the index of the nearest value in array to the given value.
        """
        return (np.abs(array - value)).argmin()

    def run(self):
        """
        Executes the full extraction procedure:
          - Loads data
          - Extracts and separates the profile
          - Computes derivatives
          - Extracts boundary layer parameters and exports a CSV file.
        """
        self.load_data()
        self.extract_profile()
        self.separate_profile()
        self.compute_profile_derivatives()
        self.write_flow_conditions()
        results = self.extract_BL_parameters_and_export()
        return results

# ===============================
# Example usage:
# ===============================
def main():
    print(f'\n{"Starting Boundary Layer Extraction":=^100}\n')
    casename = "B_10AOA_30"
    uref = 30.0
    rhoref = 1.225
    pref = 101325.0
    mu_lam = 1.78e-5
    chord = 0.3048
    alpha = 10
    input_directory = '../../'
    avbp_mean_solution = 'PostProc/Average_Field/Averaged_Solution_Reduced_Variables.h5'
    avbp_mesh = 'MESH_ZONE_Nov24/Bombardier_10AOA_Combine_Nov24.mesh.h5'
    probe_data_list = ['Group_A_Probe_Data.h5', 'Group_B_Probe_Data.h5', 'Group_C_Probe_Data.h5', 
                       'Group_D_Probe_Data.h5', 'Group_E_Probe_Data.h5', 'Group_F_Probe_Data.h5']
    #probe_data = 'Group_F_Probe_Data.h5'
    #extracted_side = 'tip'
    nb_pts = 750
    h_max = 0.015
    threshold_hmax_factor = 0.4
    chord = 0.3048
    alpha = 10
    
    for probe_data in probe_data_list:
        try:
            print(f'\n{"Processing Probe Data":=^100}\n')
            print(f'Probe Data: {probe_data}')
            # Load the probe data file
            probe_file = os.path.join(os.getcwd(), probe_data)
            # Extracting the probe data 
            exporter = ProbeDataExporter(probe_file, chord=chord, Uref=uref, alpha=alpha)
            # Export the coordinates file.
            exporter.export_coordinates()
            exporter.export_pressure_spectra()

            # Create the BoundaryLayerExtractor instance using the loaded probe data.
            extractor = BoundaryLayerExtractor(input_directory, avbp_mesh, avbp_mean_solution, probe_file ,casename,
                                                uref, rhoref, pref, mu_lam, nb_pts, h_max, threshold_hmax_factor,
                                                chord=chord, alpha=alpha)
            bl_results = extractor.run()
            print(f'\n{"Completed Boundary Layer Extraction":=^100}\n')
        except Exception as e:
            print(f"Error processing {probe_data}: {e}")

if __name__ == "__main__":
    main()