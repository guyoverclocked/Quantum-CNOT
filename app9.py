import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import gdspy

# -------------------------------
# Simulation & Geometry Parameters
# -------------------------------
three_d = False      # 3D calculation? (set False for 2D)
res = 50             # pixels/μm (high resolution for good confinement)
dpml = 2.0           # PML thickness (μm)
pad = 1.0            # Extra padding (μm)

# GDS file and layer definitions
gdsII_file = "cnot_15_march24-1.gds"
CELL_LAYER = 0
PORT1_LAYER = 1
PORT2_LAYER = 2
PORT3_LAYER = 3
PORT4_LAYER = 4
SOURCE1_LAYER = 5
UPPER_BRANCH_LAYER = 31
LOWER_BRANCH_LAYER = 32

# Material properties and thicknesses
t_oxide = 1.0
t_Si = 0.22
t_air = 0.78
cell_thickness = dpml + t_oxide + t_Si + t_air + dpml

oxide = mp.Medium(epsilon=2.25)
silicon = mp.Medium(epsilon=12)

# Simulation frequency parameters
lcen = 1.55             # wavelength in μm
fcen = 1 / lcen         # center frequency (1/μm)
df = 0.1 * fcen         # frequency width

# Z-dimension limits for (3D) simulation – here we set loose limits for 2D
si_zmax = 0.5 * t_Si if three_d else 10
si_zmin = -0.5 * t_Si if three_d else -10
cell_zmax = 0.5 * cell_thickness if three_d else 0
cell_zmin = -0.5 * cell_thickness if three_d else 0

# -------------------------------
# GDS File Loading & Processing
# -------------------------------
def load_gds_polygons(gds_file, layer=0):
    """Load polygons from GDS file."""
    print(f"Loading GDS file: {gds_file}")
    gdsii = gdspy.GdsLibrary()
    gdsii.read_gds(gds_file)
    inspect_gds(gdsii)
    
    top_cell = gdsii.top_level()[0]
    print(f"Processing top cell: {top_cell.name}")
    
    polygons = []
    shapely_polygons = []
    all_polys = top_cell.get_polygons(by_spec=True)
    
    for layer_and_polys in all_polys.items():
        current_layer = layer_and_polys[0][0]
        if current_layer == layer:
            for points in layer_and_polys[1]:
                polygons.append(points)
                shapely_polygons.append(Polygon(points))
    
    if not polygons:
        raise ValueError(f"No polygons found in layer {layer}")
        
    print(f"Loaded {len(polygons)} polygons from layer {layer}")
    return polygons, shapely_polygons

def inspect_gds(gdsii):
    """Inspect and print GDS file layers."""
    top_cell = gdsii.top_level()[0]
    polys = top_cell.get_polygons(by_spec=True)
    print("Available layers:", [spec[0] for spec in polys.keys()])
    for spec, points in polys.items():
        print(f"Layer {spec[0]}: {len(points)} polygons")

gds_file = "cnot_15_march24-1.gds"
polygons, shapely_polygons = load_gds_polygons(gds_file, layer=0)

def eps_func(r):
    """Define the spatially varying dielectric constant."""
    x, y = r.x, r.y
    for poly in shapely_polygons:
        if poly.contains(Point(x, y)):
            return 12.0  # Silicon region
    return 1.0         # Air outside the waveguides

# -------------------------------
# Determine Simulation Bounds
# -------------------------------
all_points = np.array([pt for poly in polygons for pt in poly])
xmin = np.min(all_points[:, 0])
xmax = np.max(all_points[:, 0])
ymin = np.min(all_points[:, 1])
ymax = np.max(all_points[:, 1])

# Ensure the simulation cell includes both the sources and the entire structure
# Use the actual coordinates from the GDS visualization
src_x = -4.6395  # Source x position from GDS analysis
det_x = 4.1605   # Detector x position from GDS analysis

# Adjust bounds to include sources and detectors if needed
xmin = min(xmin, src_x - 1.0)
xmax = max(xmax, det_x + 1.0) 

# Add padding
pad = 2.0  # Extra padding in μm
sx = (xmax - xmin) + 2 * pad
sy = (ymax - ymin) + 2 * pad
cell_size = mp.Vector3(sx, sy, 0)

# Calculate center position for the simulation cell
cell_center_x = (xmin + xmax) / 2
cell_center_y = (ymin + ymax) / 2

# -------------------------------
# Define Source Positions for |00⟩ State
# -------------------------------
# For the |00⟩ case we need the control photon in waveguide C₀ (2nd waveguide)
# and the target photon in waveguide T₀ (4th waveguide).
# Get positions from GDS visualization
control_src_center = mp.Vector3(-4.6395, -2.3797, 0)  # Control |0⟩ waveguide position
target_src_center  = mp.Vector3(-4.6395, -0.8507, 0)  # Target |0⟩ waveguide position

# -------------------------------
# Create Two EigenMode Sources
# -------------------------------
# Both sources use the same Gaussian source profile and propagation direction.
sources = [
    mp.EigenModeSource(
        src=mp.GaussianSource(frequency=fcen, fwidth=df),
        size=mp.Vector3(0, 2.0),
        center=control_src_center,
        eig_band=1,
        eig_parity=mp.EVEN_Y + mp.ODD_Z,
        eig_match_freq=True,
        eig_kpoint=mp.Vector3(1, 0, 0),
        direction=mp.X,
        amplitude=1.0
    ),
    mp.EigenModeSource(
        src=mp.GaussianSource(frequency=fcen, fwidth=df),
        size=mp.Vector3(0, 2.0),
        center=target_src_center,
        eig_band=1,
        eig_parity=mp.EVEN_Y + mp.ODD_Z,
        eig_match_freq=True,
        eig_kpoint=mp.Vector3(1, 0, 0),
        direction=mp.X,
        amplitude=1.0
    )
]

# -------------------------------
# Load Geometry from GDS Layers
# -------------------------------
# Load the main cell and branches from their respective layers.
cell = mp.GDSII_vol(gdsII_file, CELL_LAYER, cell_zmin, cell_zmax)
upper_branch = mp.get_GDSII_prisms(silicon, gdsII_file, UPPER_BRANCH_LAYER, si_zmin, si_zmax)
lower_branch = mp.get_GDSII_prisms(silicon, gdsII_file, LOWER_BRANCH_LAYER, si_zmin, si_zmax)
geometry = upper_branch + lower_branch

# Optionally add an oxide layer (if running a 3D simulation)
if three_d:
    oxide_center = mp.Vector3(z=-0.25 * t_oxide)
    oxide_size = mp.Vector3(cell.size.x, cell.size.y, t_oxide)
    oxide_layer = [mp.Block(material=oxide, center=oxide_center, size=oxide_size)]
    geometry = geometry + oxide_layer

# -------------------------------
# Simulation Setup
# -------------------------------
sim = mp.Simulation(
    resolution=res,
    cell_size=mp.Vector3(
        round(sx * res) / res,
        round(sy * res) / res,
        0
    ),
    boundary_layers=[mp.PML(dpml, direction=mp.X),
                     mp.PML(dpml, direction=mp.Y)],
    sources=sources,
    geometry=geometry,
    force_complex_fields=True,
    eps_averaging=True
)

# Shift the simulation origin to center on the GDS structure
sim.geometry_center = mp.Vector3(cell_center_x, cell_center_y, 0)

# -------------------------------
# Visualization Routine
# -------------------------------
def output_fields(sim):
    """Plot the Ez field overlaying the dielectric structure."""
    eps_data = sim.get_epsilon()
    ez_data = np.real(sim.get_efield_z())
    
    plt.figure(figsize=(12, 8), dpi=200)
    
    # Calculate extent based on simulation geometry center
    cell_center = sim.geometry_center
    cell_size = sim.cell_size
    extent = [
        cell_center.x - cell_size.x/2, 
        cell_center.x + cell_size.x/2,
        cell_center.y - cell_size.y/2,
        cell_center.y + cell_size.y/2
    ]
    
    # Plot dielectric structure
    plt.imshow(np.transpose(eps_data), 
               interpolation="spline36",
               cmap="binary",
               vmin=1.0,
               vmax=12.0,
               origin='lower',
               extent=extent)
    
    # Plot Ez field with dynamic color scaling
    field_max = max(abs(np.min(ez_data)), abs(np.max(ez_data)))
    plt.imshow(np.transpose(ez_data),
               interpolation="spline36",
               cmap="RdBu",
               alpha=0.9,
               vmin=-field_max/2,
               vmax=field_max/2,
               origin='lower',
               extent=extent)
    
    # Mark source positions
    plt.scatter(control_src_center.x, control_src_center.y, color='green', marker='o', s=50, label='Control Source')
    plt.scatter(target_src_center.x, target_src_center.y, color='magenta', marker='o', s=50, label='Target Source')
    
    # Add legend and grid
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    
    plt.axis('equal')
    plt.colorbar(label='Ez Field')
    plt.title(f"t = {sim.meep_time():.1f}")
    
    # Save temporary images for GIF and the final frame
    if sim.meep_time() >= sim.fields.t - 10:
        plt.savefig("final_field.png", bbox_inches='tight', pad_inches=0)
    plt.savefig(f"temp_field_{sim.meep_time():06.1f}.png", bbox_inches='tight', pad_inches=0)
    plt.close()

# -------------------------------
# Run the Simulation
# -------------------------------
sim.run(mp.at_every(5, output_fields), until=300)

# -------------------------------
# Create GIF of Field Evolution
# -------------------------------
import glob
import os
from PIL import Image

temp_files = sorted(glob.glob('temp_field_*.png'))
frames = [Image.open(f) for f in temp_files]
frames[0].save(
    'field_evolution.gif',
    save_all=True,
    append_images=frames[1:],
    duration=100,
    loop=0
)