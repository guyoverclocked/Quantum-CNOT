import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import gdspy

# Simulation parameters
# res = 25  # pixels/μm
three_d = False  # 3d calculation?
d = 0.29  # branch separation
res = 50  # Increased resolution for better confinement
dpml = 2.0  # Increased PML thickness
pad = 1.0 

# GDS file layers
gdsII_file = "cnot_15_march24-1.gds"
CELL_LAYER = 0
PORT1_LAYER = 1
PORT2_LAYER = 2
PORT3_LAYER = 3
PORT4_LAYER = 4
SOURCE1_LAYER = 5
UPPER_BRANCH_LAYER = 31
LOWER_BRANCH_LAYER = 32

# Material properties
t_oxide = 1.0
t_Si = 0.22
t_air = 0.78
cell_thickness = dpml + t_oxide + t_Si + t_air + dpml

oxide = mp.Medium(epsilon=2.25)
silicon = mp.Medium(epsilon=12)

# Simulation frequency
lcen = 1.55  # wavelength (um)
fcen = 1/lcen  # Center frequency (wavelength = 1.55 μm)
df = 0.1 * fcen  # Narrower frequency width for better mode confinement

# Z-dimension limits
si_zmax = 0.5 * t_Si if three_d else 10
si_zmin = -0.5 * t_Si if three_d else -10
cell_zmax = 0.5 * cell_thickness if three_d else 0
cell_zmin = -0.5 * cell_thickness if three_d else 0

# Load and process GDS file
def load_gds_polygons(gds_file, layer=0):
    """Load polygons from GDS file"""
    print(f"Loading GDS file: {gds_file}")
    gdsii = gdspy.GdsLibrary()
    gdsii.read_gds(gds_file)
    inspect_gds(gdsii)
    
    # Get top cell
    top_cell = gdsii.top_level()[0]
    print(f"Processing top cell: {top_cell.name}")
    
    # Extract polygons from specified layer
    polygons = []
    shapely_polygons = []
    
    # Get all polygons from the cell
    all_polys = top_cell.get_polygons(by_spec=True)
    
    for layer_and_polys in all_polys.items():
        current_layer = layer_and_polys[0][0]  # Extract layer number
        if current_layer == layer:
            for points in layer_and_polys[1]:
                polygons.append(points)
                shapely_polygons.append(Polygon(points))
    
    if not polygons:
        raise ValueError(f"No polygons found in layer {layer}")
        
    print(f"Loaded {len(polygons)} polygons from layer {layer}")
    return polygons, shapely_polygons

# Add this after reading the GDS file to inspect its contents
def inspect_gds(gdsii):
    top_cell = gdsii.top_level()[0]
    polys = top_cell.get_polygons(by_spec=True)
    print("Available layers:", [spec[0] for spec in polys.keys()])
    for spec, points in polys.items():
        print(f"Layer {spec[0]}: {len(points)} polygons")

# Load polygons from your GDS file
gds_file = "cnot_15_march24-1.gds"  # Update with your GDS filename
polygons, shapely_polygons = load_gds_polygons(gds_file, layer=0)

def eps_func(r):
    """Define the dielectric structure"""
    x, y = r.x, r.y
    for poly in shapely_polygons:
        if poly.contains(Point(x, y)):
            return 12.0  # Silicon
    return 1.0  # Air

# Calculate simulation bounds from polygons
all_points = np.array([pt for poly in polygons for pt in poly])
xmin = np.min(all_points[:, 0])
xmax = np.max(all_points[:, 0])
ymin = np.min(all_points[:, 1])
ymax = np.max(all_points[:, 1])
pad = 2.0  # padding in microns

# Simulation control parameters
RESOLUTION = 30          # pixels per micron
WAVELENGTH = 1.55       # microns
PAD_SIZE = 2.0         # padding in microns
SIM_TIME = 200         # simulation time
OUTPUT_INTERVAL = 10   # steps between outputs

# Source parameters
SOURCE_WIDTH = 0.5     # width of source in microns
SOURCE_OFFSET = 1.0    # distance from edge in microns

# Simulation cell size
sx = (xmax - xmin) + 2 * pad
sy = (ymax - ymin) + 2 * pad
cell_size = mp.Vector3(sx, sy, 0)

# Simulation parameters
resolution = 30  # pixels per micron
wavelength = 1.55  # microns
frequency = 1/wavelength

# Modified source position and parameters
# Place source at the leftmost waveguide input
source_y = ymin + sy/3  # Adjust y-position to match input waveguide
source_position = mp.Vector3(xmin + pad/2, source_y, 0)
source_size = mp.Vector3(0, 1.0, 0)  # Narrow source width to match waveguide

source = mp.Source(
    src=mp.ContinuousSource(frequency=frequency),
    component=mp.Ez,
    center=source_position,
    size=source_size
)

# PML layers and simulation setup
pml_layers = [mp.PML(1.0)]
sim = mp.Simulation(
    cell_size=cell_size,
    resolution=resolution,
    sources=[source],
    epsilon_func=eps_func,
    boundary_layers=pml_layers
)

# Load geometry from GDS
cell = mp.GDSII_vol(gdsII_file, CELL_LAYER, cell_zmin, cell_zmax)
p1 = mp.GDSII_vol(gdsII_file, PORT1_LAYER, si_zmin, si_zmax)
src_vol = mp.GDSII_vol(gdsII_file, SOURCE1_LAYER, si_zmin, si_zmax)

# Load waveguide geometry
upper_branch = mp.get_GDSII_prisms(silicon, gdsII_file, UPPER_BRANCH_LAYER, si_zmin, si_zmax)
lower_branch = mp.get_GDSII_prisms(silicon, gdsII_file, LOWER_BRANCH_LAYER, si_zmin, si_zmax)
geometry = upper_branch + lower_branch

# Add oxide layer for 3D
if three_d:
    oxide_center = mp.Vector3(z=-0.25 * t_oxide)
    oxide_size = mp.Vector3(cell.size.x, cell.size.y, t_oxide)
    oxide_layer = [mp.Block(material=oxide, center=oxide_center, size=oxide_size)]
    geometry = geometry + oxide_layer

# Update source parameters
sources = [
    mp.EigenModeSource(
        src=mp.GaussianSource(frequency=fcen, fwidth=df),  # Changed back to Gaussian
        size=mp.Vector3(0, 2.0),  # Match waveguide mode size
        center=src_vol.center,
        eig_band=1,
        eig_parity=mp.EVEN_Y + mp.ODD_Z,
        eig_match_freq=True,
        eig_kpoint=mp.Vector3(1,0,0),  # Force forward propagation
        direction=mp.X,  # Specify propagation direction
        amplitude=1.0
    )
]

# Update simulation parameters
sim = mp.Simulation(
    resolution=res,
    cell_size=mp.Vector3(
        round(cell.size.x * res) / res,
        round(cell.size.y * res) / res,
         0
    ),
    boundary_layers=[mp.PML(dpml, direction=mp.X),  # Directional PML
                    mp.PML(dpml, direction=mp.Y)],
    sources=sources,
    geometry=geometry,
    force_complex_fields=True,
    eps_averaging=True  # Better handling of material interfaces
)

def output_fields(sim):
    """Plot Ez field and epsilon using matplotlib"""
    eps_data = sim.get_epsilon()
    ez_data = np.real(sim.get_efield_z())
    
    plt.figure(figsize=(12,8), dpi=200)
    
    # Plot structure with better contrast
    plt.imshow(np.transpose(eps_data), 
              interpolation="spline36",  # Changed interpolation
              cmap="binary",
              vmin=1.0,
              vmax=12.0)
    
    # Plot field with improved color scaling
    field_max = max(abs(np.min(ez_data)), abs(np.max(ez_data)))
    plt.imshow(np.transpose(ez_data),
              interpolation="spline36",
              cmap="RdBu",
              alpha=0.9,
              vmin=-field_max/2,  # Dynamic color scaling
              vmax=field_max/2)
    
    plt.axis("off")
    plt.title(f"t = {sim.meep_time():.1f}")
    
    # Save only for GIF and final frame
    if sim.meep_time() >= sim.fields.t - 10:
        plt.savefig("final_field.png", bbox_inches='tight', pad_inches=0)
    plt.savefig(f"temp_field_{sim.meep_time():06.1f}.png", bbox_inches='tight', pad_inches=0)
    plt.close()

# Run simulation with shorter time steps
sim.run(mp.at_every(5, output_fields), until=300)  # Reduced simulation time, increased output frequency

# Create GIF and clean up temporary files
import glob
import os
from PIL import Image

# Create GIF
temp_files = sorted(glob.glob('temp_field_*.png'))
frames = [Image.open(f) for f in temp_files]
frames[0].save(
    'field_evolution.gif',
    save_all=True,
    append_images=frames[1:],
    duration=100,
    loop=0
)
