import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import os

# Constants and parameters
SI_N = 3.48  # Silicon refractive index
SIO2_N = 1.44  # Silicon dioxide refractive index
WAVELENGTH_UM = 1.55  # Operating wavelength
RESOLUTION = 30  # Pixels per micron
PML_THICKNESS = 1.0  # PML thickness in microns
PADDING = 2.0  # Padding around the device in microns

# GDSII parameters
GDSII_FILE = "final/cnot_gate_detailedspec_v6_pitchio_newwidths.gds"
SI_GDS_LAYER = 1
SI_GDS_DATATYPE = 0

# Device dimensions
DEVICE_LENGTH_UM = 6.8  # Total device length
WG_PITCH_UM = 0.8  # Center-to-center spacing between waveguides
NUM_WAVEGUIDES = 6  # Total number of waveguides

# Source and monitor parameters
SRC_OFFSET_UM = 0.5  # Distance from GDSII start to source
MON_OFFSET_UM = 0.5  # Distance from GDSII end to monitor

def setup_simulation():
    """Set up the Meep simulation environment."""
    try:
        # Calculate simulation cell size
        cell_size = mp.Vector3(
            DEVICE_LENGTH_UM + 2 * (PML_THICKNESS + PADDING),
            2 * (NUM_WAVEGUIDES * WG_PITCH_UM/2 + PML_THICKNESS + PADDING),
            0
        )
        
        # Define materials
        si = mp.Medium(epsilon=SI_N**2)
        sio2 = mp.Medium(epsilon=SIO2_N**2)
        
        # Import GDSII geometry
        geometry = mp.get_GDSII_prisms(
            si,
            GDSII_FILE,
            SI_GDS_LAYER,
            SI_GDS_DATATYPE
        )
        
        # Set up simulation
        sim = mp.Simulation(
            cell_size=cell_size,
            resolution=RESOLUTION,
            boundary_layers=[mp.PML(PML_THICKNESS)],
            geometry=geometry,
            default_material=sio2
        )
        
        return sim
    except Exception as e:
        print(f"Error in setup_simulation: {str(e)}")
        raise

def calculate_s_parameters(sim, port_index):
    """Calculate S-parameters for a specific input port."""
    try:
        # Calculate source position
        src_x = SRC_OFFSET_UM
        src_y = (port_index - (NUM_WAVEGUIDES-1)/2) * WG_PITCH_UM
        
        # Create source
        source = mp.EigenModeSource(
            src=mp.GaussianSource(frequency=1/WAVELENGTH_UM, fwidth=0.1/WAVELENGTH_UM),
            center=mp.Vector3(src_x, src_y, 0),
            size=mp.Vector3(0, 2*WG_PITCH_UM, 0),
            direction=mp.X,
            eig_band=1,
            eig_parity=mp.ODD_Z
        )
        
        # Add source to simulation
        sim.sources = [source]
        
        # Set up flux monitors for all output ports
        flux_monitors = []
        for i in range(NUM_WAVEGUIDES):
            mon_x = DEVICE_LENGTH_UM - MON_OFFSET_UM
            mon_y = (i - (NUM_WAVEGUIDES-1)/2) * WG_PITCH_UM
            flux_monitors.append(
                sim.add_flux(
                    frequency=1/WAVELENGTH_UM,
                    nfreq=1,
                    center=mp.Vector3(mon_x, mon_y, 0),
                    size=mp.Vector3(0, 2*WG_PITCH_UM, 0),
                    direction=mp.X
                )
            )
        
        # Run simulation
        sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(src_x, src_y, 0), 1e-4))
        
        # Get flux values
        s_parameters = []
        for monitor in flux_monitors:
            flux = mp.get_fluxes(monitor)[0]
            s_parameters.append(np.sqrt(flux))
        
        return s_parameters
    except Exception as e:
        print(f"Error in calculate_s_parameters for port {port_index}: {str(e)}")
        raise

def main():
    """Main function to run the S-parameter calculation."""
    try:
        # Create results directory if it doesn't exist
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Initialize S-matrix
        s_matrix = np.zeros((NUM_WAVEGUIDES, NUM_WAVEGUIDES), dtype=complex)
        
        # Calculate S-parameters for each input port
        for i in range(NUM_WAVEGUIDES):
            print(f"Calculating S-parameters for input port {i+1}...")
            sim = setup_simulation()
            s_parameters = calculate_s_parameters(sim, i)
            s_matrix[i, :] = s_parameters
            
            # Save intermediate results
            np.save(results_dir / f"s_parameters_port_{i+1}.npy", s_parameters)
        
        # Save complete S-matrix
        np.save(results_dir / "s_matrix.npy", s_matrix)
        
        # Plot S-matrix magnitude
        plt.figure(figsize=(10, 8))
        plt.imshow(np.abs(s_matrix), cmap='viridis')
        plt.colorbar(label='|S|')
        plt.title('S-Matrix Magnitude')
        plt.xlabel('Output Port')
        plt.ylabel('Input Port')
        plt.savefig(results_dir / "s_matrix_magnitude.png")
        plt.close()
        
        print("S-parameter calculation completed. Results saved in 'results' directory.")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
