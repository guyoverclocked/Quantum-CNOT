import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import sys

def run_cnot_simulation(input_state, resolution=30):
    """
    Run a simulation of the CNOT gate for a specific input state.
    
    Parameters:
    -----------
    input_state : str
        One of '00', '01', '10', or '11' to specify the input state
    resolution : int
        Simulation resolution in pixels per μm
    
    Returns:
    --------
    dict
        Results of the simulation including flux measurements
    """
    print(f"\n\n{'='*80}")
    print(f"RUNNING SIMULATION FOR INPUT STATE |{input_state}⟩")
    print(f"{'='*80}\n")
    
    # Define simulation parameters
    wavelength = 1.55         # Operating wavelength in μm (telecom wavelength)
    freq = 1/wavelength       # Frequency in μm^-1
    silicon_eps = 12.0        # Silicon dielectric constant (approximate)

    # Simulation cell size
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

    # Define GDS file path
    gds_file = "cnot_15_march24-1.gds"

    # Load geometry from GDS file
    geometry = []
    if os.path.exists(gds_file):
        try:
            # Import the geometry from GDS file
            geometry = mp.get_GDSII_prisms(mp.Medium(epsilon=silicon_eps), gds_file, 0, -1.0)
            print(f"Successfully imported geometry from {gds_file}")
            
            # Define waveguide positions for GDS-based geometry
            gate_center = mp.Vector3(0, 0, 0)
            input_x = gate_center.x - waveguide_length/2 - 2.0
            output_x = gate_center.x + waveguide_length/2 + 2.0
            
        except Exception as e:
            print(f"Error importing GDS file: {e}")
            print("Cannot continue without proper geometry.")
            sys.exit(1)
    else:
        print(f"GDS file {gds_file} not found.")
        print("Cannot continue without GDS file.")
        sys.exit(1)

    # Define waveguide positions
    # We now have estimated positions from the GDS file analysis
    wg_y_positions = [
        -3.2971,  # Waveguide 1 (Aux1)
        -2.3797,  # Waveguide 2 (Co - Control |0⟩)
        -2.0739,  # Waveguide 3 (C₁ - Control |1⟩)
        -0.8507,  # Waveguide 4 (To - Target part 1)
        -0.2391,  # Waveguide 5 (T₁ - Target part 2)
        0.0667,   # Waveguide 6 (Aux2)
    ]
    
    # Coupling region parameters (from GDS analysis)
    gate_center = mp.Vector3(-0.2395, -0.1350, 0)
    coupling_width = 4.8  # From the paper (4.8 μm)
    input_x = -2.6395     # Left edge of coupling region
    output_x = 2.1605     # Right edge of coupling region
    
    # Source and detector positions (from GDS analysis)
    src_x = -4.6395       # Source position (2 μm before coupling region)
    det_x = 4.1605        # Detector position (2 μm after coupling region)
    
    # Map waveguide indices to their purposes
    aux1_y = wg_y_positions[0]      # Auxiliary 1
    c0_y = wg_y_positions[1]        # Control |0⟩ (Co)
    c1_y = wg_y_positions[2]        # Control |1⟩ (C₁)
    t0_y = wg_y_positions[3]        # Target |0⟩ part 1 (To)
    t1_y = wg_y_positions[4]        # Target |0⟩ part 2 (T₁)
    aux2_y = wg_y_positions[5]      # Auxiliary 2
    
    # Source parameters
    pulse_width = 0.5               # Temporal width
    cutoff = 8.0                    # How many widths to simulate
    
    # Define sources based on input state
    sources = []
    
    # Control qubit source
    if input_state[0] == '0':
        # Control |0⟩: photon in waveguide Co (waveguide #2)
        sources.append(mp.Source(
            mp.GaussianSource(freq, fwidth=freq/10, is_integrated=True),
            component=mp.Ez,
            center=mp.Vector3(src_x, c0_y, 0),
            size=mp.Vector3(0, waveguide_width, 0)
        ))
    else:  # input_state[0] == '1'
        # Control |1⟩: photon in waveguide C₁ (waveguide #3)
        sources.append(mp.Source(
            mp.GaussianSource(freq, fwidth=freq/10, is_integrated=True),
            component=mp.Ez,
            center=mp.Vector3(src_x, c1_y, 0),
            size=mp.Vector3(0, waveguide_width, 0)
        ))
    
    # Target qubit source
    if input_state[1] == '0':
        # Target |0⟩: photon in superposition 1/√2(To + T₁)
        # First part of superposition: To
        sources.append(mp.Source(
            mp.GaussianSource(freq, fwidth=freq/10, is_integrated=True),
            component=mp.Ez,
            center=mp.Vector3(src_x, t0_y, 0),
            size=mp.Vector3(0, waveguide_width, 0),
            amplitude=1/np.sqrt(2)
        ))
        # Second part of superposition: T₁
        sources.append(mp.Source(
            mp.GaussianSource(freq, fwidth=freq/10, is_integrated=True),
            component=mp.Ez,
            center=mp.Vector3(src_x, t1_y, 0),
            size=mp.Vector3(0, waveguide_width, 0),
            amplitude=1/np.sqrt(2)
        ))
    else:  # input_state[1] == '1'
        # Target |1⟩: photon in superposition 1/√2(To - T₁)
        # First part of superposition: To
        sources.append(mp.Source(
            mp.GaussianSource(freq, fwidth=freq/10, is_integrated=True),
            component=mp.Ez,
            center=mp.Vector3(src_x, t0_y, 0),
            size=mp.Vector3(0, waveguide_width, 0),
            amplitude=1/np.sqrt(2)
        ))
        # Second part of superposition: T₁ (negative phase)
        sources.append(mp.Source(
            mp.GaussianSource(freq, fwidth=freq/10, is_integrated=True),
            component=mp.Ez,
            center=mp.Vector3(src_x, t1_y, 0),
            size=mp.Vector3(0, waveguide_width, 0),
            amplitude=-1/np.sqrt(2)  # Negative amplitude for |1⟩ state
        ))
    
    # Set up the simulation
    sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=sources,
        resolution=resolution
    )
    
    # Define flux regions to measure output intensities
    flux_x = det_x                  # Position to measure flux (after the CNOT gate)
    n_wg = 6                        # Number of waveguides
    
    # Create flux monitors for each waveguide output
    flux_regions = []
    flux_results = []
    
    for i in range(n_wg):
        wg_y = wg_y_positions[i]
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
    run_time = 100  # Run for sufficient time
    sim.run(until=run_time)
    
    # Retrieve the field data for visualization
    ez_data = sim.get_array(center=mp.Vector3(), size=cell_size, component=mp.Ez)
    
    # Calculate flux measurements
    fluxes = []
    for i in range(n_wg):
        fluxes.append(mp.get_fluxes(flux_results[i])[0])
    
    # Create names for the waveguides
    wg_names = [
        "Auxiliary 1",
        "Co (Control |0⟩)",
        "C₁ (Control |1⟩)",
        "To (Target part 1)",
        "T₁ (Target part 2)",
        "Auxiliary 2"
    ]
    
    # Total flux
    total_flux = sum(fluxes)
    
    # Normalize fluxes
    normalized_fluxes = [flux / total_flux for flux in fluxes]
    
    # Generate results dictionary
    results = {
        'input_state': input_state,
        'ez_data': ez_data,
        'cell_size': cell_size,
        'fluxes': fluxes,
        'normalized_fluxes': normalized_fluxes,
        'total_flux': total_flux,
        'wg_names': wg_names,
        'wg_positions': wg_y_positions,
        'flux_x': flux_x,
        'src_x': src_x
    }
    
    # Print flux measurements
    print("\nOutput Flux Measurements:")
    for i in range(n_wg):
        print(f"Waveguide {i+1} ({wg_names[i]}): {fluxes[i]:.6f}")
    
    print(f"\nTotal Flux: {total_flux:.6f}")
    
    # Print normalized fluxes
    print("\nNormalized Output Probabilities:")
    for i in range(n_wg):
        print(f"Waveguide {i+1} ({wg_names[i]}): {normalized_fluxes[i]:.6f}")
    
    # Analyze results for truth table verification
    analyze_cnot_results(results)
    
    # Plot and save the field distribution
    plot_field_distribution(results)
    
    return results

def analyze_cnot_results(results):
    """
    Analyze the simulation results to verify CNOT truth table
    """
    input_state = results['input_state']
    normalized_fluxes = results['normalized_fluxes']
    wg_names = results['wg_names']
    
    # Extract relevant indices
    c0_idx = 1  # Control |0⟩ (Co)
    c1_idx = 2  # Control |1⟩ (C₁)
    t0_idx = 3  # Target part 1 (To)
    t1_idx = 4  # Target part 2 (T₁)
    
    # Calculate control and target state probabilities
    control_0_prob = normalized_fluxes[c0_idx]
    control_1_prob = normalized_fluxes[c1_idx]
    
    # For target state, we need to determine if it's in |0⟩ or |1⟩ by looking
    # at the relative phase between To and T₁
    target_0_part = normalized_fluxes[t0_idx]
    target_1_part = normalized_fluxes[t1_idx]
    
    # Total probability in the target waveguides
    target_total = target_0_part + target_1_part
    
    # Check if target is properly distributed for |0⟩ state (equal amplitudes)
    target_is_0 = (target_total > 0) and (abs(target_0_part - target_1_part) / target_total < 0.3)
    
    # Output control state
    control_output = '0' if control_0_prob > control_1_prob else '1'
    
    # Output target state (simplified determination)
    # In a real quantum system, we would look at the phase relationship
    # Here we're using amplitude differences as an approximation
    target_output = '0' if target_is_0 else '1'
    
    # Expected output based on CNOT truth table
    expected_output = input_state[0]  # Control bit should stay the same
    if input_state[0] == '1':
        # If control is 1, target bit flips
        expected_output += '1' if input_state[1] == '0' else '0'
    else:
        # If control is 0, target bit stays the same
        expected_output += input_state[1]
    
    actual_output = control_output + target_output
    
    print(f"\nQCNOT Gate Truth Table Verification (|{input_state}⟩ → |{expected_output}⟩):")
    print(f"Measured output state: |{actual_output}⟩")
    print(f"Control state: {control_output} (expected: {expected_output[0]})")
    print(f"Target state: {target_output} (expected: {expected_output[1]})")
    print(f"Truth table match: {'Yes' if actual_output == expected_output else 'No'}")
    
    # Calculate fidelity (simplified)
    control_fidelity = (control_0_prob if expected_output[0] == '0' else control_1_prob)
    print(f"Control Fidelity: {control_fidelity:.4f}")
    
    # Display auxiliary waveguide leakage
    aux_leakage = normalized_fluxes[0] + normalized_fluxes[5]
    print(f"Auxiliary Waveguide Leakage: {aux_leakage:.4f}")

def plot_field_distribution(results):
    """
    Plot the field distribution from the simulation results
    """
    input_state = results['input_state']
    ez_data = results['ez_data']
    cell_size = results['cell_size']
    wg_names = results['wg_names']
    wg_positions = results['wg_positions']
    flux_x = results['flux_x']
    
    # Plot the field distribution
    plt.figure(figsize=(12, 6))
    sx, sy = cell_size.x, cell_size.y
    
    custom_cmap = LinearSegmentedColormap.from_list('custom_colormap', 
                                                  [(0, 'blue'), (0.5, 'white'), (1, 'red')])
    plt.imshow(ez_data.transpose(), interpolation='spline16', cmap=custom_cmap,
             origin='lower', extent=[-sx/2, sx/2, -sy/2, sy/2])
    plt.colorbar(label='Ez Field')
    plt.title(f'QCNOT Gate Simulation - |{input_state}⟩ Input State')
    plt.xlabel('x (μm)')
    plt.ylabel('y (μm)')
    
    # Mark the waveguide positions
    for i in range(len(wg_positions)):
        wg_y = wg_positions[i]
        plt.text(flux_x - 0.5, wg_y, wg_names[i], color='white', ha='right', va='center',
               bbox=dict(facecolor='black', alpha=0.7))
    
    # Save the figure
    plt.savefig(f"qcnot_{input_state}_state_simulation.png", dpi=300, bbox_inches='tight')
    print(f"Field distribution saved as 'qcnot_{input_state}_state_simulation.png'")
    plt.close()

if __name__ == "__main__":
    # Run simulations for all four input states
    input_states = ['00', '01', '10', '11']
    
    # If command line arguments are provided, use them as input states
    if len(sys.argv) > 1:
        input_states = [sys.argv[1]]
    
    # Lower resolution for faster simulations
    resolution = 20
    
    # Results for each state
    all_results = {}
    
    for state in input_states:
        all_results[state] = run_cnot_simulation(state, resolution)
    
    # Print final summary
    print("\n\n" + "="*80)
    print("CNOT GATE SIMULATION SUMMARY")
    print("="*80)
    
    # Print truth table comparison
    print("\nTruth Table Verification:")
    print("-----------------------")
    print("Input | Expected | Measured | Match")
    print("-----------------------")
    
    for state in all_results:
        results = all_results[state]
        
        # Extract control and target output
        c0_idx, c1_idx = 1, 2
        t0_idx, t1_idx = 3, 4
        
        norm_fluxes = results['normalized_fluxes']
        
        # Determine measured output state
        control_out = '0' if norm_fluxes[c0_idx] > norm_fluxes[c1_idx] else '1'
        target_is_0 = abs(norm_fluxes[t0_idx] - norm_fluxes[t1_idx]) / (norm_fluxes[t0_idx] + norm_fluxes[t1_idx]) < 0.3
        target_out = '0' if target_is_0 else '1'
        
        measured = control_out + target_out
        
        # Calculate expected output
        expected = state[0]  # Control bit should stay the same
        if state[0] == '1':
            # If control is 1, target bit flips
            expected += '1' if state[1] == '0' else '0'
        else:
            # If control is 0, target bit stays the same
            expected += state[1]
        
        match = "Yes" if measured == expected else "No"
        
        print(f"|{state}⟩   |   |{expected}⟩    |   |{measured}⟩    |   {match}")
    
    print("-----------------------")
    print("\nSimulation complete. Results saved as PNG files.") 