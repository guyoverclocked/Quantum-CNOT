import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import sys

def run_single_photon_simulation(waveguide_idx, phase=0, resolution=20):
    """
    Run a single-photon simulation through a specific waveguide with a given phase.
    
    Parameters:
    -----------
    waveguide_idx : int
        Index of the waveguide to inject the photon (0-5)
    phase : float
        Phase of the photon (in radians)
    resolution : int
        Simulation resolution in pixels per μm
    
    Returns:
    --------
    dict
        Results of the simulation including field distribution and fluxes
    """
    print(f"\nRunning single-photon simulation for waveguide {waveguide_idx+1} with phase {phase}")
    
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

    # Load geometry from GDS file
    gds_file = "cnot_15_march24-1.gds"
    geometry = []
    
    if not os.path.exists(gds_file):
        print(f"GDS file {gds_file} not found.")
        sys.exit(1)
        
    try:
        # Import the geometry from GDS file
        geometry = mp.get_GDSII_prisms(mp.Medium(epsilon=silicon_eps), gds_file, 0, -1.0)
    except Exception as e:
        print(f"Error importing GDS file: {e}")
        sys.exit(1)

    # Waveguide y-positions (from GDS analysis)
    wg_y_positions = [
        -3.2971,  # Waveguide 1 (Aux1)
        -2.3797,  # Waveguide 2 (Co - Control |0⟩)
        -2.0739,  # Waveguide 3 (C₁ - Control |1⟩)
        -0.8507,  # Waveguide 4 (To - Target part 1)
        -0.2391,  # Waveguide 5 (T₁ - Target part 2)
        0.0667,   # Waveguide 6 (Aux2)
    ]
    
    # Coupling region parameters (from GDS analysis)
    central_x = -0.2395
    central_y = -0.1350
    coupling_width = 4.8
    
    # Source and detector positions
    src_x = -4.6395          # Source position (2 μm before coupling region)
    det_x = 4.1605           # Detector position (2 μm after coupling region)
    
    # Waveguide parameters
    waveguide_width = 0.45   # Width of waveguides (450 nm)
    
    # Source parameters
    source_y = wg_y_positions[waveguide_idx]
    amplitude = np.exp(1j * phase)  # Complex amplitude with phase
    
    # Define source
    sources = [
        mp.Source(
            mp.GaussianSource(freq, fwidth=freq/10, is_integrated=True),
            component=mp.Ez,
            center=mp.Vector3(src_x, source_y, 0),
            size=mp.Vector3(0, waveguide_width, 0),
            amplitude=amplitude
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
    
    # Define flux regions to measure output intensities and phases
    flux_x = det_x            # Position to measure flux
    n_wg = 6                  # Number of waveguides
    
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
    
    # Get complex amplitudes at each detector
    complex_amplitudes = []
    for i in range(n_wg):
        # Get the complex amplitudes from the flux objects
        flux = mp.get_fluxes(flux_results[i])[0]
        # The phase information isn't directly accessible in the flux
        # We would need a more sophisticated approach to get true phase
        # This is a simplification for demonstration
        complex_amplitudes.append(flux)
    
    # Calculate flux measurements (power)
    fluxes = []
    for i in range(n_wg):
        fluxes.append(mp.get_fluxes(flux_results[i])[0])
    
    # Total flux
    total_flux = sum(fluxes)
    
    # Normalize fluxes
    normalized_fluxes = [flux / total_flux for flux in fluxes]
    
    # Create names for the waveguides
    wg_names = [
        "Auxiliary 1",
        "Co (Control |0⟩)",
        "C₁ (Control |1⟩)",
        "To (Target part 1)",
        "T₁ (Target part 2)",
        "Auxiliary 2"
    ]
    
    # Generate results dictionary
    results = {
        'input_waveguide': waveguide_idx,
        'input_phase': phase,
        'ez_data': ez_data,
        'fluxes': fluxes,
        'normalized_fluxes': normalized_fluxes,
        'complex_amplitudes': complex_amplitudes,
        'total_flux': total_flux,
        'wg_names': wg_names,
        'wg_positions': wg_y_positions
    }
    
    # Print normalized fluxes
    print("\nOutput Probabilities:")
    for i in range(n_wg):
        print(f"  WG{i+1} ({wg_names[i]}): {normalized_fluxes[i]:.6f}")
    
    return results

def approximate_two_photon_interference(control_results, target_results):
    """
    Approximate the two-photon quantum interference by combining single-photon simulations.
    
    This is a simplification of the actual quantum behavior, but can give insight
    into the expected output state of the CNOT gate.
    
    Parameters:
    -----------
    control_results : dict
        Results of the control photon simulation
    target_results : dict
        Results of the target photon simulation
    
    Returns:
    --------
    dict
        Combined results approximating two-photon interference
    """
    # Extract the normalized fluxes (probability amplitudes)
    control_fluxes = control_results['normalized_fluxes']
    target_fluxes = target_results['normalized_fluxes']
    
    # The actual two-photon interference would involve a more complex quantum mechanical
    # calculation accounting for the Hong-Ou-Mandel effect and other quantum phenomena.
    # Here we use a simple approximation:
    
    # Calculate joint probabilities for finding one photon in each output waveguide
    joint_probabilities = np.zeros((6, 6))
    
    for i in range(6):  # Control output waveguide
        for j in range(6):  # Target output waveguide
            # Simple product of probabilities (ignoring true quantum interference)
            joint_probabilities[i, j] = control_fluxes[i] * target_fluxes[j]
    
    # Extract output state probabilities based on waveguide mapping
    # Control state probabilities (waveguides 2-3)
    c0_prob = joint_probabilities[1, :].sum()  # Probability of control in |0⟩ (Co)
    c1_prob = joint_probabilities[2, :].sum()  # Probability of control in |1⟩ (C₁)
    
    # Target state probabilities (waveguides 4-5)
    # For |0⟩ target state, we expect balanced presence in To and T₁
    # For |1⟩ target state, we expect To and T₁ with phase difference
    t0_to_prob = joint_probabilities[:, 3].sum()  # Probability in To
    t0_t1_prob = joint_probabilities[:, 4].sum()  # Probability in T₁
    
    # Simplified analysis for target state
    # If flux is balanced between To and T₁, it's approximately |0⟩ state
    # If flux is predominantly in one, it's approximately |1⟩ state
    target_balance = min(t0_to_prob, t0_t1_prob) / max(t0_to_prob, t0_t1_prob)
    target_is_0 = target_balance > 0.7  # If reasonably balanced, consider it |0⟩
    
    # Determine the measured output state
    control_out = '0' if c0_prob > c1_prob else '1'
    target_out = '0' if target_is_0 else '1'
    measured_state = control_out + target_out
    
    # Calculate auxiliary waveguide leakage
    aux_leakage = (
        joint_probabilities[0, :].sum() +  # Aux1 output
        joint_probabilities[5, :].sum() +  # Aux2 output
        joint_probabilities[:, 0].sum() +  # Aux1 output
        joint_probabilities[:, 5].sum()    # Aux2 output
    ) / 2  # Avoid double counting
    
    # Generate combined results
    combined_results = {
        'joint_probabilities': joint_probabilities,
        'control_0_prob': c0_prob,
        'control_1_prob': c1_prob,
        'target_to_prob': t0_to_prob,
        'target_t1_prob': t0_t1_prob,
        'target_balance': target_balance,
        'target_is_0': target_is_0,
        'measured_state': measured_state,
        'aux_leakage': aux_leakage
    }
    
    return combined_results

def simulate_cnot_truth_table(resolution=20):
    """
    Simulate the CNOT gate truth table by running simulations for all input states.
    
    Parameters:
    -----------
    resolution : int
        Simulation resolution in pixels per μm
    """
    print("\n" + "="*80)
    print("CNOT GATE TRUTH TABLE SIMULATION (QUANTUM APPROXIMATION)")
    print("="*80)
    
    # Define the input states to test
    input_states = [
        {'name': '00', 'control_wg': 1, 'control_phase': 0, 'target_wg1': 3, 'target_wg2': 4, 'target_phase1': 0, 'target_phase2': 0},
        {'name': '01', 'control_wg': 1, 'control_phase': 0, 'target_wg1': 3, 'target_wg2': 4, 'target_phase1': 0, 'target_phase2': np.pi},
        {'name': '10', 'control_wg': 2, 'control_phase': 0, 'target_wg1': 3, 'target_wg2': 4, 'target_phase1': 0, 'target_phase2': 0},
        {'name': '11', 'control_wg': 2, 'control_phase': 0, 'target_wg1': 3, 'target_wg2': 4, 'target_phase1': 0, 'target_phase2': np.pi}
    ]
    
    # Results for each state
    all_results = {}
    
    for state in input_states:
        print("\n" + "="*80)
        print(f"SIMULATING INPUT STATE |{state['name']}⟩")
        print("="*80)
        
        # Simulate control photon
        control_results = run_single_photon_simulation(
            state['control_wg'], 
            phase=state['control_phase'],
            resolution=resolution
        )
        
        # Simulate target photon part 1
        target1_results = run_single_photon_simulation(
            state['target_wg1'],
            phase=state['target_phase1'],
            resolution=resolution
        )
        
        # Simulate target photon part 2
        target2_results = run_single_photon_simulation(
            state['target_wg2'],
            phase=state['target_phase2'],
            resolution=resolution
        )
        
        # Combine the target photon results (superposition)
        # In a real quantum simulation, this would be a coherent superposition
        # Here we use a simplified approach
        target_results = {
            'normalized_fluxes': [
                0.5 * (target1_results['normalized_fluxes'][i] + target2_results['normalized_fluxes'][i])
                for i in range(6)
            ]
        }
        
        # Approximate two-photon interference
        combined_results = approximate_two_photon_interference(control_results, target_results)
        
        # Calculate expected output based on CNOT truth table
        input_state = state['name']
        expected_output = input_state[0]  # Control bit stays the same
        
        if input_state[0] == '1':
            # If control is 1, target bit flips
            expected_output += '1' if input_state[1] == '0' else '0'
        else:
            # If control is 0, target bit stays the same
            expected_output += input_state[1]
        
        # Print the results
        print("\nCNOT Gate Truth Table Verification:")
        print(f"  Input state: |{input_state}⟩")
        print(f"  Expected output: |{expected_output}⟩")
        print(f"  Measured output: |{combined_results['measured_state']}⟩")
        print(f"  Truth table match: {'Yes' if combined_results['measured_state'] == expected_output else 'No'}")
        print(f"  Control state fidelity: {max(combined_results['control_0_prob'], combined_results['control_1_prob']):.4f}")
        print(f"  Auxiliary waveguide leakage: {combined_results['aux_leakage']:.4f}")
        
        # Store the results
        all_results[input_state] = {
            'control_results': control_results,
            'target_results': target_results,
            'combined_results': combined_results,
            'expected_output': expected_output
        }
    
    # Print truth table summary
    print("\n" + "="*80)
    print("CNOT GATE TRUTH TABLE SUMMARY")
    print("="*80)
    print("\nTruth Table Verification:")
    print("-----------------------")
    print("Input | Expected | Measured | Match")
    print("-----------------------")
    
    for state_name in ['00', '01', '10', '11']:
        if state_name in all_results:
            results = all_results[state_name]
            expected = results['expected_output']
            measured = results['combined_results']['measured_state']
            match = "Yes" if measured == expected else "No"
            
            print(f"|{state_name}⟩   |   |{expected}⟩    |   |{measured}⟩    |   {match}")
    
    print("-----------------------")
    
    return all_results

if __name__ == "__main__":
    # Run the CNOT truth table simulation with a lower resolution for speed
    simulate_cnot_truth_table(resolution=15) 