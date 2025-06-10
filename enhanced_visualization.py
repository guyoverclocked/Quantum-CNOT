import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Simulation parameters
resolution = 60           # Higher resolution for better wave patterns
wavelength = 1.55         # Operating wavelength in μm
fcen = 1/wavelength       # Center frequency
df = 0.2 * fcen          # Match reference code frequency width
silicon_eps = 12.0        # Silicon dielectric constant

# GDS file path
gds_file = "cnot_15_march24-1.gds"

def run_simulation():
    """Run a CNOT gate simulation for the |00⟩ state with visualization matching the reference image"""
    print("Running CNOT gate simulation for |00⟩ state...")
    
    # Check if GDS file exists
    if not os.path.exists(gds_file):
        print(f"GDS file {gds_file} not found. Cannot continue.")
        return None, None, None
    
    # Cell dimensions based on the actual device
    sx = 16  # Cell width in µm
    sy = 10  # Cell height in µm
    cell_size = mp.Vector3(sx, sy, 0)
    
    # PML layers
    pml_layers = [mp.PML(1.0)]
    
    # Waveguide parameters
    waveguide_width = 0.45  # Waveguide width in µm (450 nm)
    
    # Waveguide y-positions from reference image
    # These positions align with where the waveguides appear in the reference image
    wg_y_positions = [
        -3.2,    # Waveguide 1 (Aux1)
        -2.5,    # Waveguide 2 (Co - Control |0⟩)
        -1.3,    # Waveguide 3 (C₁ - Control |1⟩)
        -0.6,    # Waveguide 4 (To - Target part 1)
        0.6,     # Waveguide 5 (T₁ - Target part 2)
        1.3,     # Waveguide 6 (Aux2)
    ]
    
    # Waveguide names
    wg_names = [
        "Auxiliary 1",
        "Co (Control |0⟩)",
        "C₁ (Control |1⟩)",
        "To (Target part 1)",
        "T₁ (Target part 2)",
        "Auxiliary 2"
    ]
    
    # Import geometry from GDS
    geometry = mp.get_GDSII_prisms(mp.Medium(epsilon=silicon_eps), gds_file, 0, -1.0)
    print(f"Imported geometry from {gds_file}")
    
    # Source and detector positions
    src_x = -6.0
    det_x = 4.0
    
    # For |00⟩ state, only add sources to Control |0⟩ (WG2) and Target |0⟩ (WG4)
    control_idx = 1  # Control |0⟩ waveguide (WG2)
    target_idx = 3   # Target |0⟩ waveguide (WG4)
    
    # Define sources - using continuous sources
    sources = [
        # Control |0⟩ source
        mp.Source(
            mp.ContinuousSource(frequency=fcen, width=df),
            component=mp.Ez,
            center=mp.Vector3(src_x, wg_y_positions[control_idx], 0),
            size=mp.Vector3(0, waveguide_width, 0),
            amplitude=1.0
        ),
        # Target |0⟩ source
        mp.Source(
            mp.ContinuousSource(frequency=fcen, width=df),
            component=mp.Ez,
            center=mp.Vector3(src_x, wg_y_positions[target_idx], 0),
            size=mp.Vector3(0, waveguide_width, 0),
            amplitude=1.0
        )
    ]
    
    # Set up simulation
    sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=sources,
        resolution=resolution,
        force_complex_fields=True
    )
    
    # Add flux monitors to measure output
    flux_results = []
    
    for y_pos in wg_y_positions:
        flux_region = mp.FluxRegion(
            center=mp.Vector3(det_x, y_pos, 0),
            size=mp.Vector3(0, waveguide_width, 0)
        )
        flux_results.append(
            sim.add_flux(fcen, 0, 1, flux_region)
        )
    
    # Run simulation
    sim.run(until=110)
    
    # Get dielectric and field data directly from simulation
    eps_data = sim.get_epsilon()
    ez_data = sim.get_array(center=mp.Vector3(), size=cell_size, component=mp.Ez)
    
    # Convert complex field data to real values
    if np.iscomplexobj(ez_data):
        print("Converting complex field data to real values...")
        ez_data = np.real(ez_data)
    
    # Calculate flux values
    fluxes = [mp.get_fluxes(fr)[0] for fr in flux_results]
    total_flux = sum(fluxes) if sum(fluxes) != 0 else 1.0
    normalized_fluxes = [f/total_flux for f in fluxes]
    
    # Create visualization to match the reference
    create_exact_reference_visualization(eps_data, ez_data, wg_y_positions, control_idx, target_idx)
    
    return fluxes, normalized_fluxes, wg_names

def create_exact_reference_visualization(eps_data, ez_data, wg_y_positions, control_idx, target_idx):
    """Create visualization exactly like the reference image, using the actual structure from the GDS file"""
    
    # Get dimensions
    sx, sy = eps_data.shape
    
    # Create figure with dimensions matching reference image
    plt.figure(figsize=(16, 9), dpi=300)
    
    # Set extent to match reference image
    extent = [-8, 8, -5, 5]  # Spatial extent matching reference
    
    # STEP 1: Draw the structure directly from simulation data (no rectangles)
    # This shows the exact waveguide structure from the GDS file
    
    # Create binary structure mask - thresholding the epsilon data
    structure_mask = (eps_data > 1.1)  # Silicon has epsilon > 1
    
    # Plot pure white background first
    plt.imshow(
        np.ones_like(structure_mask.T),  # White background
        cmap='gray',
        vmin=0,
        vmax=1,
        extent=extent,
        origin='lower',
        interpolation='nearest',
        aspect='auto'
    )
    
    # Plot the structure (waveguides) in light gray
    plt.imshow(
        structure_mask.T * 0.8,  # Light gray (0.8) for the waveguides
        cmap='gray',
        vmin=0,
        vmax=1,
        extent=extent,
        origin='lower',
        interpolation='nearest',
        aspect='auto',
        alpha=1.0
    )
    
    # STEP 2: Mask the field data to only show inside active waveguides
    # Using the actual structure mask for accurate field confinement
    
    # Create field masks for the active waveguides only
    # First get the y-position ranges for active waveguides
    active_y_positions = [wg_y_positions[control_idx], wg_y_positions[target_idx]]
    waveguide_width = 0.45
    
    # Create masks for each active waveguide
    active_masks = []
    for y_pos in active_y_positions:
        # Convert to pixel coordinates (from -5 to 5 in y direction)
        y_min_idx = int((y_pos - waveguide_width/2 + 5) * sy / 10)
        y_max_idx = int((y_pos + waveguide_width/2 + 5) * sy / 10)
        # Create row mask for this waveguide
        row_mask = np.zeros(sy, dtype=bool)
        row_mask[y_min_idx:y_max_idx] = True
        # Only keep fields where both row mask AND structure mask are True
        mask_2d = np.zeros_like(structure_mask, dtype=bool)
        for x in range(sx):
            for y in range(sy):
                if row_mask[y] and structure_mask[x, y]:
                    mask_2d[x, y] = True
        active_masks.append(mask_2d)
    
    # Combine masks for all active waveguides
    field_mask = np.zeros_like(structure_mask, dtype=bool)
    for mask in active_masks:
        field_mask = np.logical_or(field_mask, mask)
    
    # Apply mask to field data
    masked_ez = np.zeros_like(ez_data)
    masked_ez[field_mask] = ez_data[field_mask]
    
    # Scale field data for vibrant colors
    max_val = np.max(np.abs(masked_ez))
    if max_val > 0:
        # Stronger scaling for deeper colors like reference image
        scaled_ez = masked_ez / (max_val * 0.3)
    else:
        scaled_ez = masked_ez
    
    # Plot field data with the same colormap and scaling as reference
    plt.imshow(
        scaled_ez.T,
        cmap='RdBu_r',  # Red-Blue colormap as in reference
        vmin=-1,
        vmax=1,
        extent=extent,
        origin='lower',
        interpolation='bilinear',  # Smoother interpolation for waves
        alpha=1.0,
        aspect='auto'
    )
    
    # Add title and formatting
    plt.title(f"t = 110.0", fontsize=16)
    plt.axis('off')
    
    # Save high-quality image
    plt.tight_layout()
    plt.savefig("cnot_00_enhanced_visualization.png", dpi=300, bbox_inches='tight', pad_inches=0)
    print("Enhanced visualization saved as 'cnot_00_enhanced_visualization.png'")
    plt.close()

if __name__ == "__main__":
    # Run simulation and create the visualization
    fluxes, normalized_fluxes, wg_names = run_simulation()
    
    # Print results if available
    if fluxes:
        print("\nOutput Flux Measurements:")
        for i, (name, flux, norm_flux) in enumerate(zip(wg_names, fluxes, normalized_fluxes)):
            print(f"Waveguide {i+1} ({name}): {flux:.6f} ({norm_flux*100:.2f}%)") 