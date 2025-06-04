# module/probe_exporter.py

import os
import csv
import numpy as np
import h5py
import re
from scipy.signal import welch

class ProbeDataExporter:
    """
    A class to export probe data from an HDF5 file into two types of CSV files
    (space-delimited):
    
    1. T<alpha>_BL_<group>_Coordinates.csv:
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
       and saved in a file named "T<alpha>_WPS_A<sensor_tag_without_space>_Spectrum.csv".
    """
    
    def __init__(self, probe_filepath, chord:float=0.3048, Uref:float=30.0, alpha:int=10, LE_dist:float=1.245, density:bool = True):
        """
        Parameters:
            probe_filepath: Path to the HDF5 file containing the probe data.
            chord: The chord length (default 0.3048 m).
            Uref: The reference (free-stream) velocity (default 30 m/s).
            alpha: Angle of attack in degrees (default 10).
            LE_dist: Distance from the leading edge to the probe (default 1.245 m).
        """
        print(f'\n{"Beginning Probe Data Extraction":.^60}\n')
        self.probe_filepath = probe_filepath
        self.chord = chord
        self.Uref = Uref
        self.alpha = alpha
        self.LE_dist = LE_dist
        self.density = density
        match = re.search(r'Group_([A-Z])_Probe', probe_filepath)  # Searches for 'Group_A_Probe' pattern.
        if match:
            self.group_letter = match.group(1)
        new_dir = os.path.join(os.getcwd(),"T"+f"{self.alpha:02d}_Group_"+self.group_letter)
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
        named "T<alpha>_BL_<group>_Coordinates.csv" with the following 9 columns:
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
                zBL = 1 - max((np.abs(zBL) - 0.1034) / self.chord, 0) if self.group_letter not in ['A','B'] else 0.0
                # For this example, we assume the WPS data are the same as the BL data.
                idx_str = f"{i:03d}"
                row = [idx_str, xBL, yBL, zBL, idx_str, xBL, yBL, zBL, 0]
                rows.append(row)
        fmt = "{:03d} {:10.8f} {:10.8f} {:10.8f} {:03d} {:10.8f} {:10.8f} {:10.8f} {:2.4f} "  # Adjust width as needed
        filename = "T" + self.group_letter + f"{self.alpha:02d}_coordinates.csv"
        csv_filename = os.path.join(self.export_dir, filename)
        if os.path.exists(csv_filename):
            os.remove(csv_filename)
        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f, delimiter=" ")
            header = ["idxBL", "xBL", "yBL", "zBL", "idxWPS", "xWPS", "yWPS", "zWPS", "dist"]
            writer.writerow(header)
            for row in rows:
                f.write(fmt.format(int(row[0]), row[1], row[2], row[3], int(row[4]), row[5], row[6], row[7], row[8])+ "\n")  # Fixed-width formatting
        print("    CSV coordinate file written:",  os.path.basename(os.path.normpath(csv_filename))) 
    
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
                scaling_factor = - 10*np.log10(np.mean(np.diff(freq))) if self.density else 1
                PSD_dB = 10 * np.log10(PSD / (p_ref**2))  + scaling_factor
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
                print("    CSV WPS file written:",  os.path.basename(os.path.normpath(csv_filename))) 
        print(f'\n{"Complete Probe Data Extraction":.^60}\n')
