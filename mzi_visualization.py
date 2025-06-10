import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

def run_mzi_simulation(resolution=25):
    """
    Run a simulation of a basic Mach-Zehnder Interferometer (MZI),
    which is a fundamental building block of quantum photonic gates.
    """
    print("Running Mach-Zehnder Interferometer (MZI) simulation...")
    
    # Define simulation parameters
    wavelength = 1.55         # Operating wavelength in μm (telecom wavelength)
    freq = 1/wavelength       # Frequency in μm^-1
    silicon_eps = 12.0        # Silicon dielectric constant (approximate)

    # Simulation cell size
    sx = 10                   # Cell size in x-direction (μm)
    sy = 6                    # Cell size in y-direction (μm)
    cell_size = mp.Vector3(sx, sy, 0)

    # PML layers to absorb radiation at boundaries
    pml_thickness = 1.0
    pml_layers = [mp.PML(pml_thickness)]

    # Waveguide parameters
    waveguide_width = 0.45    # Width of waveguides (450 nm)
    waveguide_gap = 1.5       # Vertical gap between waveguides (1.5 μm)
    arm_length = 3.0          # Length of MZI arms (3.0 μm)
    bend_radius = 1.0         # Bend radius for waveguide curves
    
    # Define geometry: simple MZI structure
    geometry = []
    
    # Function to add a curved waveguide segment
    def add_arc_segment(x_center, y_center, radius, angle_start, angle_end, width, material):
        # Create a curved waveguide segment using multiple small blocks
        n_points = 20
        angles = np.linspace(angle_start, angle_end, n_points)
        for i in range(n_points-1):
            angle_mid = (angles[i] + angles[i+1]) / 2
            x_mid = x_center + radius * np.cos(angle_mid)
            y_mid = y_center + radius * np.sin(angle_mid)
            
            dx = radius * (np.cos(angles[i+1]) - np.cos(angles[i]))
            dy = radius * (np.sin(angles[i+1]) - np.sin(angles[i]))
            
            # Determine the rotation angle
            block_angle = np.arctan2(dy, dx)
            
            # Determine segment length
            segment_length = np.sqrt(dx**2 + dy**2)
            
            geometry.append(mp.Block(
                center=mp.Vector3(x_mid, y_mid, 0),
                size=mp.Vector3(segment_length, width, mp.inf),
                e1=mp.Vector3(np.cos(block_angle), np.sin(block_angle), 0),
                e2=mp.Vector3(-np.sin(block_angle), np.cos(block_angle), 0),
                material=material
            ))
    
    # Input waveguide 
    geometry.append(mp.Block(
        center=mp.Vector3(-sx/3 - 1, 0, 0),
        size=mp.Vector3(sx/3, waveguide_width, mp.inf),
        material=mp.Medium(epsilon=silicon_eps)
    ))
    
    # First beam splitter (50:50)
    geometry.append(mp.Block(
        center=mp.Vector3(-sx/6, 0, 0),
        size=mp.Vector3(waveguide_width, waveguide_width*2, mp.inf),
        material=mp.Medium(epsilon=silicon_eps)
    ))
    
    # Top arm of the MZI
    # - First curve up
    add_arc_segment(-sx/6 + waveguide_width/2 + bend_radius, waveguide_width/2,
                  bend_radius, -np.pi/2, 0, waveguide_width,
                  mp.Medium(epsilon=silicon_eps))
    
    # - Straight top segment
    top_y = waveguide_width/2 + bend_radius
    geometry.append(mp.Block(
        center=mp.Vector3(0, top_y, 0),
        size=mp.Vector3(arm_length, waveguide_width, mp.inf),
        material=mp.Medium(epsilon=silicon_eps)
    ))
    
    # - Second curve down
    add_arc_segment(arm_length/2 + bend_radius, waveguide_width/2 + bend_radius,
                  bend_radius, np.pi, np.pi*3/2, waveguide_width,
                  mp.Medium(epsilon=silicon_eps))
    
    # Bottom arm of the MZI
    # - First curve down
    add_arc_segment(-sx/6 + waveguide_width/2 + bend_radius, -waveguide_width/2,
                  bend_radius, np.pi/2, np.pi, waveguide_width,
                  mp.Medium(epsilon=silicon_eps))
    
    # - Straight bottom segment
    bottom_y = -(waveguide_width/2 + bend_radius)
    geometry.append(mp.Block(
        center=mp.Vector3(0, bottom_y, 0),
        size=mp.Vector3(arm_length, waveguide_width, mp.inf),
        material=mp.Medium(epsilon=silicon_eps)
    ))
    
    # - Second curve up
    add_arc_segment(arm_length/2 + bend_radius, -(waveguide_width/2 + bend_radius),
                  bend_radius, 0, np.pi/2, waveguide_width,
                  mp.Medium(epsilon=silicon_eps))
    
    # Second beam splitter (50:50)
    geometry.append(mp.Block(
        center=mp.Vector3(sx/6, 0, 0),
        size=mp.Vector3(waveguide_width, waveguide_width*2, mp.inf),
        material=mp.Medium(epsilon=silicon_eps)
    ))
    
    # Output waveguide
    geometry.append(mp.Block(
        center=mp.Vector3(sx/3 + 1, 0, 0),
        size=mp.Vector3(sx/3, waveguide_width, mp.inf),
        material=mp.Medium(epsilon=silicon_eps)
    ))
    
    # Source position (input of MZI)
    src_x = -sx/2 + pml_thickness + 1.0
    src_y = 0
    
    # Create a single-frequency source
    sources = [
        mp.Source(
            mp.GaussianSource(freq, fwidth=freq/10, is_integrated=True),
            component=mp.Ez,
            center=mp.Vector3(src_x, src_y, 0),
            size=mp.Vector3(0, waveguide_width, 0)
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
    
    # Define flux regions to measure output
    flux_x = sx/2 - pml_thickness - 1.0  # Output position
    
    # Create flux monitor for output
    flux_region = mp.FluxRegion(
        center=mp.Vector3(flux_x, 0, 0),
        size=mp.Vector3(0, waveguide_width, 0)
    )
    flux_result = sim.add_flux(freq, 0, 1, flux_region)
    
    # Run the simulation
    run_time = 80  # Run for sufficient time
    sim.run(until=run_time)
    
    # Get field data for visualization
    eps_data = sim.get_array(center=mp.Vector3(), size=cell_size, component=mp.Dielectric)
    ez_data = sim.get_array(center=mp.Vector3(), size=cell_size, component=mp.Ez)
    
    # Calculate flux
    output_flux = mp.get_fluxes(flux_result)[0]
    
    # Create visualization
    create_mzi_visualization(eps_data, ez_data, cell_size, src_x, flux_x, 
                           silicon_eps, output_flux, top_y, bottom_y)
    
    return output_flux

def create_mzi_visualization(eps_data, ez_data, cell_size, src_x, flux_x, 
                           silicon_eps, output_flux, top_y, bottom_y):
    """
    Create a comprehensive visualization of the MZI simulation.
    """
    # Convert to spatial coordinates
    sx, sy = cell_size.x, cell_size.y
    x = np.linspace(-sx/2, sx/2, eps_data.shape[0])
    y = np.linspace(-sy/2, sy/2, eps_data.shape[1])
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 1, height_ratios=[1, 3, 1])
    
    # 1. Title area
    ax_title = fig.add_subplot(gs[0])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, 'Mach-Zehnder Interferometer (MZI) Simulation', 
                 ha='center', va='center', fontsize=20, fontweight='bold')
    
    # 2. Main field visualization
    ax_main = fig.add_subplot(gs[1])
    
    # First show the waveguide structure (dielectric)
    eps_plot = ax_main.imshow(eps_data.transpose(), cmap='binary', 
                             origin='lower', extent=[-sx/2, sx/2, -sy/2, sy/2],
                             alpha=0.5, vmax=silicon_eps*1.2)
    
    # Overlay the field visualization
    custom_cmap = LinearSegmentedColormap.from_list('custom_colormap', 
                                                  [(0, 'blue'), (0.5, 'white'), (1, 'red')])
    field_plot = ax_main.imshow(ez_data.transpose(), cmap=custom_cmap, 
                              origin='lower', extent=[-sx/2, sx/2, -sy/2, sy/2],
                              alpha=0.8)
    
    # Add colorbar for the field
    cbar = plt.colorbar(field_plot, ax=ax_main)
    cbar.set_label('Electric Field (Ez)')
    
    # Mark key points on the MZI
    # Source
    ax_main.scatter(src_x, 0, color='green', s=100, marker='*', 
                  label='Input Source')
    ax_main.text(src_x, 0.5, 'Source', ha='center', va='bottom', color='green')
    
    # First beam splitter
    ax_main.text(-sx/6, -1, 'First Beam Splitter\n(50:50)', ha='center', va='top', 
               color='blue', bbox=dict(facecolor='white', alpha=0.7))
    
    # Phase shift region (for future reference)
    ax_main.text(0, top_y+0.5, 'Top Arm', ha='center', va='bottom', 
               color='purple')
    ax_main.text(0, bottom_y-0.5, 'Bottom Arm', ha='center', va='top', 
               color='purple')
    
    # Second beam splitter
    ax_main.text(sx/6, -1, 'Second Beam Splitter\n(50:50)', ha='center', va='top', 
               color='blue', bbox=dict(facecolor='white', alpha=0.7))
    
    # Detector
    ax_main.scatter(flux_x, 0, color='red', s=100, marker='D', 
                  label='Output Detector')
    ax_main.text(flux_x, 0.5, 'Detector', ha='center', va='bottom', color='red')
    
    # Add legend
    ax_main.legend(loc='upper right')
    
    # Add axis labels
    ax_main.set_xlabel('x (μm)')
    ax_main.set_ylabel('y (μm)')
    ax_main.set_title('Field Propagation in Mach-Zehnder Interferometer')
    
    # 3. Explanation area
    ax_explain = fig.add_subplot(gs[2])
    ax_explain.axis('off')
    
    # Explanation text
    explanation_text = (
        "Mach-Zehnder Interferometer (MZI) Principles:\n\n"
        "1. The input light enters from the left and encounters the first beam splitter\n"
        "2. The beam splitter creates a superposition state, sending 50% of the light to each arm\n"
        "3. Light traveling through the top and bottom arms accumulates different phases\n"
        "4. At the second beam splitter, the light from both arms interferes\n"
        "5. Constructive or destructive interference determines the output intensity\n\n"
        f"Measured Output Intensity: {output_flux:.6f}\n\n"
        "The MZI is the building block for more complex photonic quantum gates like the CNOT gate.\n"
        "In a CNOT gate implementation, multiple MZIs are arranged to create quantum interference effects."
    )
    
    ax_explain.text(0.5, 0.5, explanation_text, ha='center', va='center', 
                   fontsize=12, linespacing=1.5, 
                   bbox=dict(facecolor='lightyellow', alpha=0.5, boxstyle='round'))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('mzi_visualization.png', dpi=300, bbox_inches='tight')
    print("MZI visualization saved as 'mzi_visualization.png'")
    
    plt.show()

if __name__ == "__main__":
    # Run the MZI simulation and visualization
    output_flux = run_mzi_simulation(resolution=30)
    print(f"MZI Output Flux: {output_flux:.6f}") 