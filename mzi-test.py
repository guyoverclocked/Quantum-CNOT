import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

# Define simulation parameters based on the research paper dimensions
resolution = 30           # Pixels per μm (higher for better accuracy)
wavelength = 1.55         # Operating wavelength in μm (telecom wavelength)
freq = 1/wavelength       # Frequency in μm^-1
silicon_eps = 12.0        # Silicon dielectric constant (approximate)

# Simulation cell size (slightly larger than the CNOT gate dimensions)
sx = 16                   # Cell size in x-direction (μm)
sy = 10                   # Cell size in y-direction (μm)
cell_size = mp.Vector3(sx, sy, 0)

# PML layers to absorb radiation at boundaries
pml_thickness = 1.0
pml_layers = [mp.PML(pml_thickness)]

# Waveguide parameters
waveguide_width = 0.45    # Width of waveguides (450 nm)
waveguide_gap = 0.2       # Gap between waveguides (200 nm)
waveguide_length = 4.8    # CNOT region length from paper (4.8 μm)

# Define GDS file path (replace with actual path if available)
gds_file = "cnot_15_march24-1.gds"

# If GDS file is available, use it; otherwise, create a simplified geometry
geometry = []
use_simplified = False
if os.path.exists(gds_file):
    try:
        # Import the geometry from GDS file
        geometry = mp.get_GDSII_prisms(mp.Medium(epsilon=silicon_eps), gds_file, 0, -1.0)
        print(f"Successfully imported geometry from {gds_file}")
        
        # Define waveguide positions for GDS-based geometry
        # These positions should match the GDS file layout
        gate_center = mp.Vector3(0, 0, 0)
        waveguide_length = 4.8    # CNOT region length from paper (4.8 μm)
        input_x = gate_center.x - waveguide_length/2
        output_x = gate_center.x + waveguide_length/2
        
    except Exception as e:
        print(f"Error importing GDS file: {e}")
        # Fall back to simplified geometry creation
        use_simplified = True
else:
    print(f"GDS file {gds_file} not found, using simplified geometry.")
    use_simplified = True

# Create simplified geometry if GDS import failed or file not found
if use_simplified:
    # Create six waveguides approximating the CNOT structure described in paper
    gate_center = mp.Vector3(0, 0, 0)
    
    # Position of the leftmost point of the CNOT region
    input_x = gate_center.x - waveguide_length/2
    
    # Position of the rightmost point of the CNOT region
    output_x = gate_center.x + waveguide_length/2
    
    # Create the six waveguides of the CNOT gate
    for i in range(6):
        y_pos = -1.25 + i * (waveguide_width + waveguide_gap)
        
        # Input waveguide segment
        geometry.append(mp.Block(
            size=mp.Vector3(input_x + 7, waveguide_width, mp.inf),
            center=mp.Vector3(input_x/2 - 3.5, y_pos, 0),
            material=mp.Medium(epsilon=silicon_eps)
        ))
        
        # CNOT region (coupling region)
        geometry.append(mp.Block(
            size=mp.Vector3(waveguide_length, waveguide_width, mp.inf),
            center=mp.Vector3(gate_center.x, y_pos, 0),
            material=mp.Medium(epsilon=silicon_eps)
        ))
        
        # Output waveguide segment
        geometry.append(mp.Block(
            size=mp.Vector3(sx - output_x - pml_thickness - 1, waveguide_width, mp.inf),
            center=mp.Vector3(output_x + (sx - output_x - pml_thickness - 1)/2, y_pos, 0),
            material=mp.Medium(epsilon=silicon_eps)
        ))
    
    print("Created simplified six-waveguide CNOT geometry")

# Index mapping for waveguides (as per paper)
# WG#1: Auxiliary, WG#2: Co (Control |0⟩), WG#3: C₁ (Control |1⟩)
# WG#4: To, WG#5: T₁ (Target superposition states)
# WG#6: Auxiliary

# Source parameters
src_x = input_x - 2.0  # Place sources before the CNOT region
pulse_width = 0.5      # Temporal width
cutoff = 8.0           # How many widths to simulate

# For |00⟩ state: 
# - Control photon in waveguide #2 (Co)
# - Target photon in superposition across waveguides #4-#5 (To+T₁)

# Define sources for |00⟩ input state
control_y = -1.25 + 1 * (waveguide_width + waveguide_gap)  # Waveguide #2 (Co)
target_y1 = -1.25 + 3 * (waveguide_width + waveguide_gap)  # Waveguide #4 (To)
target_y2 = -1.25 + 4 * (waveguide_width + waveguide_gap)  # Waveguide #5 (T₁)

sources = [
    # Control photon in |0⟩ state (waveguide Co)
    mp.Source(
        mp.GaussianSource(freq, fwidth=freq/10, is_integrated=True),
        component=mp.Ez,
        center=mp.Vector3(src_x, control_y, 0),
        size=mp.Vector3(0, waveguide_width, 0)
    ),
    # Target photon in superposition (1/√2)(To + T₁)
    # First part of superposition: To
    mp.Source(
        mp.GaussianSource(freq, fwidth=freq/10, is_integrated=True),
        component=mp.Ez,
        center=mp.Vector3(src_x, target_y1, 0),
        size=mp.Vector3(0, waveguide_width, 0),
        amplitude=1/np.sqrt(2)
    ),
    # Second part of superposition: T₁
    mp.Source(
        mp.GaussianSource(freq, fwidth=freq/10, is_integrated=True),
        component=mp.Ez,
        center=mp.Vector3(src_x, target_y2, 0),
        size=mp.Vector3(0, waveguide_width, 0),
        amplitude=1/np.sqrt(2)
    )
]

# Set up the simulation
sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=sources,
    resolution=resolution
)

# Define flux regions to measure output intensities
flux_x = output_x + 1.0  # Position to measure flux (after the CNOT gate)
n_wg = 6                # Number of waveguides

# Create flux monitors for each waveguide output
flux_regions = []
flux_results = []

for i in range(n_wg):
    wg_y = -1.25 + i * (waveguide_width + waveguide_gap)
    flux_regions.append(
        mp.FluxRegion(
            center=mp.Vector3(flux_x, wg_y, 0),
            size=mp.Vector3(0, waveguide_width, 0)
        )
    )
    # Add flux monitor for each waveguide
    flux_results.append(
        sim.add_flux(freq, 0, 1, flux_regions[i])
    )

# Run the simulation
sim.run(until=100)  # Run for sufficient time to allow propagation through the device

# Retrieve the field data for visualization
ez_data = sim.get_array(center=mp.Vector3(), size=cell_size, component=mp.Ez)

# Plot the field distribution
plt.figure(figsize=(12, 6))
custom_cmap = LinearSegmentedColormap.from_list('custom_colormap', 
                                                [(0, 'blue'), (0.5, 'white'), (1, 'red')])
plt.imshow(ez_data.transpose(), interpolation='spline16', cmap=custom_cmap,
           origin='lower', extent=[-sx/2, sx/2, -sy/2, sy/2])
plt.colorbar(label='Ez Field')
plt.title('QCNOT Gate Simulation - |00⟩ Input State')
plt.xlabel('x (μm)')
plt.ylabel('y (μm)')

# Mark the waveguide positions
for i in range(n_wg):
    wg_y = -1.25 + i * (waveguide_width + waveguide_gap)
    if i == 1:
        plt.text(flux_x + 0.5, wg_y, 'Co', color='white', ha='left', va='center')
    elif i == 2:
        plt.text(flux_x + 0.5, wg_y, 'C₁', color='white', ha='left', va='center')
    elif i == 3:
        plt.text(flux_x + 0.5, wg_y, 'To', color='white', ha='left', va='center')
    elif i == 4:
        plt.text(flux_x + 0.5, wg_y, 'T₁', color='white', ha='left', va='center')
    else:
        plt.text(flux_x + 0.5, wg_y, f'Aux{i+1 if i>4 else i}', color='white', ha='left', va='center')

# Add source labels
plt.text(src_x, control_y, '|0⟩ₒ', color='black', ha='center', va='center', 
         bbox=dict(facecolor='white', alpha=0.7))
plt.text(src_x, (target_y1 + target_y2)/2, '|0⟩ₜ', color='black', ha='center', va='center',
         bbox=dict(facecolor='white', alpha=0.7))

# Save the figure
plt.savefig("qcnot_00_state_simulation.png", dpi=300, bbox_inches='tight')
plt.close()

# Calculate and print flux measurements
print("\nOutput Flux Measurements:")
total_flux = 0
for i in range(n_wg):
    flux = mp.get_fluxes(flux_results[i])[0]
    total_flux += flux
    wg_name = ""
    if i == 1:
        wg_name = "Co (Control |0⟩)"
    elif i == 2:
        wg_name = "C₁ (Control |1⟩)"
    elif i == 3:
        wg_name = "To (Target part 1)"
    elif i == 4:
        wg_name = "T₁ (Target part 2)"
    else:
        wg_name = f"Auxiliary {i+1 if i>4 else i}"
    
    print(f"Waveguide {i+1} ({wg_name}): {flux:.6f}")

print(f"\nTotal Flux: {total_flux:.6f}")

# Normalize the fluxes to analyze the output state probabilities
normalized_fluxes = []
for i in range(n_wg):
    normalized_fluxes.append(mp.get_fluxes(flux_results[i])[0] / total_flux)

# For |00⟩ input, we expect the output to remain |00⟩
# So, we should see most of the flux in waveguides 2 (Co) and a superposition in 4-5 (To+T₁)
print("\nNormalized Output Probabilities:")
for i in range(n_wg):
    wg_name = ""
    if i == 1:
        wg_name = "Co (Control |0⟩)"
    elif i == 2:
        wg_name = "C₁ (Control |1⟩)"
    elif i == 3:
        wg_name = "To (Target part 1)"
    elif i == 4:
        wg_name = "T₁ (Target part 2)"
    else:
        wg_name = f"Auxiliary {i+1 if i>4 else i}"
    
    print(f"Waveguide {i+1} ({wg_name}): {normalized_fluxes[i]:.6f}")

# Calculate the fidelity of the operation for |00⟩ state
# For perfect operation, we expect all control flux in Co and target flux split between To and T₁
theoretical_control_ratio = 1.0  # All control photon flux should be in waveguide Co (wg#2)
theoretical_target_ratio = 0.5   # Target photon flux should be evenly split between To and T₁

control_fidelity = normalized_fluxes[1] / (normalized_fluxes[1] + normalized_fluxes[2])
target_fidelity = (normalized_fluxes[3] + normalized_fluxes[4] > 0) and \
                  (abs(normalized_fluxes[3]/(normalized_fluxes[3] + normalized_fluxes[4]) - 0.5) < 0.1)

print("\nQCNOT Gate Truth Table Verification (|00⟩ → |00⟩):")
print(f"Control State Fidelity: {control_fidelity:.4f}")
print(f"Target State Properly Distributed: {'Yes' if target_fidelity else 'No'}")
print(f"Overall |00⟩ State Test Result: {'Passed' if control_fidelity > 0.9 and target_fidelity else 'Failed'}")
