import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image

# Simulation parameters
resolution = 50
dpml = 2.0
lcen = 1.55  # wavelength (um)
fcen = 1/lcen
df = 0.05 * fcen  # Narrower frequency width for better mode confinement

# Material properties
silicon = mp.Medium(epsilon=12)

# Load geometry from GDS file
gdsII_file = "cnot_15_march24-1.gds"
cell_zmin = -10
cell_zmax = 10

# Layer definitions
CELL_LAYER = 0
PORT1_LAYER = 1   # Control input
PORT2_LAYER = 2   # Target input
PORT3_LAYER = 3   # Control output
PORT4_LAYER = 4   # Target output

# Function to run CNOT simulation
def run_cnot_simulation(input_config):
    control_on, target_on = input_config
    
    # Create output directory
    outdir = f"cnot_{int(control_on)}{int(target_on)}"
    os.makedirs(outdir, exist_ok=True)
    
    print(f"Running simulation for input |{int(control_on)}{int(target_on)}⟩...")
    
    # Load structure volumes from GDS
    cell = mp.GDSII_vol(gdsII_file, CELL_LAYER, cell_zmin, cell_zmax)
    control_in_vol = mp.GDSII_vol(gdsII_file, PORT1_LAYER, cell_zmin, cell_zmax)
    target_in_vol = mp.GDSII_vol(gdsII_file, PORT2_LAYER, cell_zmin, cell_zmax)
    control_out_vol = mp.GDSII_vol(gdsII_file, PORT3_LAYER, cell_zmin, cell_zmax)
    target_out_vol = mp.GDSII_vol(gdsII_file, PORT4_LAYER, cell_zmin, cell_zmax)
    
    # Load waveguide geometry
    geometry = mp.get_GDSII_prisms(silicon, gdsII_file, CELL_LAYER, cell_zmin, cell_zmax)
    
    # Setup sources for control and target qubits
    sources = []
    
    if control_on:
        control_src = mp.EigenModeSource(
            src=mp.GaussianSource(frequency=fcen, fwidth=df),
            size=control_in_vol.size,
            center=control_in_vol.center,
            eig_band=1,
            eig_parity=mp.EVEN_Y + mp.ODD_Z,
            eig_match_freq=True,
            eig_kpoint=mp.Vector3(1,0,0),
            direction=mp.X,
            amplitude=1.0
        )
        sources.append(control_src)
    
    if target_on:
        target_src = mp.EigenModeSource(
            src=mp.GaussianSource(frequency=fcen, fwidth=df),
            size=target_in_vol.size,
            center=target_in_vol.center,
            eig_band=1,
            eig_parity=mp.EVEN_Y + mp.ODD_Z,
            eig_match_freq=True,
            eig_kpoint=mp.Vector3(1,0,0),
            direction=mp.X,
            amplitude=1.0
        )
        sources.append(target_src)
    
    # Create simulation with a dummy source if no sources defined
    if not sources:
        dummy_src = mp.Source(
            mp.GaussianSource(frequency=fcen, fwidth=df),
            component=mp.Ez,
            center=mp.Vector3(),
            size=mp.Vector3(0,0,0),
            amplitude=0.0  # Zero amplitude won't affect simulation
        )
        sources = [dummy_src]
    
    # Prepare simulation
    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell.size,
        boundary_layers=[mp.PML(dpml, direction=mp.X), mp.PML(dpml, direction=mp.Y)],
        sources=sources,
        geometry=geometry,
        force_complex_fields=True,
        eps_averaging=True
    )
    
    # Add flux monitors at output ports
    control_out_mon = sim.add_flux(
        fcen, 0, 1,
        mp.FluxRegion(center=control_out_vol.center, size=control_out_vol.size)
    )
    
    target_out_mon = sim.add_flux(
        fcen, 0, 1,
        mp.FluxRegion(center=target_out_vol.center, size=target_out_vol.size)
    )
    
    def output_fields(sim):
        """Visualize fields and structure with enhanced contrast"""
        eps_data = sim.get_epsilon()
        ez_data = np.real(sim.get_efield_z())
        
        # Create figure
        plt.figure(figsize=(10, 8), dpi=200)
        
        # Plot structure with better contrast
        plt.imshow(
            np.transpose(eps_data),
            interpolation="spline36",
            cmap="binary",
            vmin=1.0,
            vmax=12.0
        )
        
        # Calculate field max for better color scaling
        field_max = max(abs(np.min(ez_data)), abs(np.max(ez_data)))
        if field_max > 0:
            plt.imshow(
                np.transpose(ez_data),
                interpolation="spline36",
                cmap="RdBu",
                alpha=0.9,
                vmin=-field_max/2,
                vmax=field_max/2
            )
        
        # Add title and annotations
        input_label = f"|{int(control_on)}{int(target_on)}⟩"
        
        # Determine expected output based on CNOT truth table
        if not control_on and not target_on:   # |00⟩
            expected = "|00⟩"
        elif not control_on and target_on:     # |01⟩
            expected = "|01⟩"
        elif control_on and not target_on:     # |10⟩
            expected = "|11⟩"
        else:                                  # |11⟩
            expected = "|10⟩"
            
        plt.title(f"Input: {input_label}, Expected: {expected}, t={sim.meep_time():.1f}")
        
        # Add annotations for input/output ports
        cell_x, cell_y = eps_data.shape[0], eps_data.shape[1]
        
        # Convert MEEP coordinates to pixel positions
        def to_pixel(v):
            x_pixel = int((v.x - sim.cell_size.x/-2) / sim.cell_size.x * cell_x)
            y_pixel = int((v.y - sim.cell_size.y/-2) / sim.cell_size.y * cell_y)
            return (x_pixel, y_pixel)
        
        # Add port labels if applicable
        if control_on:
            plt.annotate("Control In", xy=to_pixel(control_in_vol.center), 
                        color='white', fontsize=10, ha='center', va='center',
                        bbox=dict(boxstyle="round", fc="blue", alpha=0.7))
                        
        if target_on:
            plt.annotate("Target In", xy=to_pixel(target_in_vol.center), 
                        color='white', fontsize=10, ha='center', va='center',
                        bbox=dict(boxstyle="round", fc="blue", alpha=0.7))
        
        # Add output labels
        plt.annotate("Control Out", xy=to_pixel(control_out_vol.center), 
                    color='white', fontsize=10, ha='center', va='center',
                    bbox=dict(boxstyle="round", fc="green", alpha=0.7))
                    
        plt.annotate("Target Out", xy=to_pixel(target_out_vol.center), 
                    color='white', fontsize=10, ha='center', va='center',
                    bbox=dict(boxstyle="round", fc="green", alpha=0.7))
        
        plt.axis("off")
        
        # Save frame
        plt.savefig(f"{outdir}/field_{sim.meep_time():06.1f}.png", 
                  bbox_inches='tight')
        
        # Save final frame separately
        if sim.meep_time() >= sim.fields.t - 10:
            plt.savefig(f"{outdir}/final.png", bbox_inches='tight')
        
        plt.close()
    
    # Run simulation
    sim.run(mp.at_every(10, output_fields), until=300)
    
    # Get output flux values
    control_flux = mp.get_fluxes(control_out_mon)[0]
    target_flux = mp.get_fluxes(target_out_mon)[0]
    
    # Calculate probabilities
    total_flux = control_flux + target_flux
    if total_flux > 0:
        control_prob = control_flux / total_flux
        target_prob = target_flux / total_flux
    else:
        control_prob = 0
        target_prob = 0
    
    # Create GIF from output frames
    frames = [Image.open(f) for f in sorted(glob.glob(f"{outdir}/field_*.png"))]
    if frames:
        frames[0].save(
            f"{outdir}/evolution.gif",
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0
        )
    
    # Clean up temporary files
    for f in glob.glob(f"{outdir}/field_*.png"):
        if "final" not in f:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass
    
    return {
        "input": f"|{int(control_on)}{int(target_on)}⟩",
        "control_out": control_prob,
        "target_out": target_prob
    }

# Main function to verify CNOT gate truth table
def main():
    # Truth table inputs: (control, target)
    input_configs = [
        (False, True),   # |01⟩ - Start with this one since |00⟩ has no sources
        (True, False),   # |10⟩
        (True, True),    # |11⟩
        (False, False),  # |00⟩ - Do last to prevent empty field issues
    ]
    
    results = []
    for config in input_configs:
        results.append(run_cnot_simulation(config))
    
    # Display truth table
    print("\nCNOT Gate Truth Table Results:")
    print("-----------------------------")
    print("Input  | Control Out | Target Out | Expected")
    print("-----------------------------")
    
    # Reorder to standard truth table order
    results.sort(key=lambda r: r["input"])
    
    for r in results:
        input_state = r['input']
        c_prob = r['control_out']
        t_prob = r['target_out']
        
        # Determine expected output
        if input_state == "|00⟩":
            expected = "|00⟩"
        elif input_state == "|01⟩":
            expected = "|01⟩"
        elif input_state == "|10⟩":
            expected = "|11⟩"
        elif input_state == "|11⟩":
            expected = "|10⟩"
        
        print(f"{input_state}  |  {c_prob:.2f}       |    {t_prob:.2f}     |  {expected}")
    
    # Create summary visualization
    plt.figure(figsize=(12, 10))
    plt.suptitle("CNOT Gate Truth Table Verification", fontsize=16)
    
    for i, result in enumerate(results):
        plt.subplot(2, 2, i+1)
        try:
            img = plt.imread(f"cnot_{result['input'][1:-1]}/final.png")
            plt.imshow(img)
        except FileNotFoundError:
            plt.text(0.5, 0.5, f"Missing image for {result['input']}", 
                    horizontalalignment='center', fontsize=14)
        plt.title(f"Input: {result['input']}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("cnot_summary.png", dpi=200)
    print("\nCreated summary visualization in 'cnot_summary.png'")

if __name__ == "__main__":
    main()