import gdspy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Load the GDS file
print("Loading GDS file...")
gdsii = gdspy.GdsLibrary()
gdsii.read_gds("cnot_15_march24-1.gds")
print("GDS file loaded successfully.")

# Get all cells in the GDS file
cells = gdsii.cells
print(f"Found {len(cells)} cell(s) in the GDS file:")
for cell_name, cell in cells.items():
    print(f"  - Cell: {cell_name}")
    
    # Count polygons in the cell
    polygons = cell.get_polygons()
    print(f"    Contains {len(polygons)} polygon(s)")

# Get the top cell (usually the main design)
if cells:
    # Get the main cell (first one or top one)
    if "TOP" in cells:
        main_cell = cells["TOP"]
    else:
        main_cell = list(cells.values())[0]
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plot all polygons in the main cell
    polygons = main_cell.get_polygons()
    
    # Store polygon centroids to help identify waveguides
    centroids = []
    
    for poly in polygons:
        # Extract x and y coordinates, add the first point to close the polygon
        x = np.append(poly[:, 0], poly[0, 0])
        y = np.append(poly[:, 1], poly[0, 1])
        ax.plot(x, y, 'b-')
        
        # Calculate and store centroid
        centroid_x = np.mean(poly[:, 0])
        centroid_y = np.mean(poly[:, 1])
        centroids.append((centroid_x, centroid_y))
        
        # Add small dot at centroid for reference
        ax.plot(centroid_x, centroid_y, 'r.', markersize=3)
    
    # Get the bounds of the cell
    bounds = main_cell.get_bounding_box()
    if bounds is not None:
        # Add some padding
        padding = 1.0
        min_x, min_y = bounds[0] - padding
        max_x, max_y = bounds[1] + padding
        
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
    
    # Try to identify the waveguides based on the layout pattern
    # For a typical linear waveguide array, the y-coordinates will cluster around 6 distinct values
    
    # Extract all y-coordinates
    all_y = [c[1] for c in centroids]
    
    # If there are enough polygons, try to identify distinct waveguide paths
    if len(centroids) > 10:
        # Create a histogram to identify clusters of y-coordinates
        hist, bin_edges = np.histogram(all_y, bins=20)
        
        # Find the prominent y-positions (potential waveguides)
        # This is a simple approach, might need refinement for complex layouts
        prominent_bins = []
        for i, count in enumerate(hist):
            if count > 1:  # Consider bins with multiple polygons as potential waveguides
                bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
                prominent_bins.append((bin_center, count))
        
        # Sort by y-coordinate (bottom to top)
        prominent_bins.sort(key=lambda x: x[0])
        
        # Label the prominent waveguides (assuming up to 6 waveguides)
        waveguide_labels = ["Aux1", "Co (Control |0⟩)", "C₁ (Control |1⟩)", 
                            "To (Target part 1)", "T₁ (Target part 2)", "Aux2"]
        
        # Estimate waveguide y-positions (for use in simulation)
        print("\nEstimated Waveguide Positions:")
        print("-----------------------------")
        
        min_x_bound = min([c[0] for c in centroids])
        max_x_bound = max([c[0] for c in centroids])
        
        # Colors for different waveguides
        waveguide_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
        
        for i, (y_pos, count) in enumerate(prominent_bins[:6]):
            if i < len(waveguide_labels):
                label = waveguide_labels[i]
                color = waveguide_colors[i % len(waveguide_colors)]
                
                # Add a label on the right side
                ax.text(max_x_bound + 0.5, y_pos, f"WG{i+1}: {label}", 
                         ha='left', va='center', fontsize=10, color=color,
                         bbox=dict(facecolor='white', alpha=0.7))
                
                # Print the estimated y-position
                print(f"Waveguide {i+1} ({label}): y ≈ {y_pos:.4f}")
                
                # Mark the waveguide path with a horizontal dashed line
                ax.plot([min_x_bound, max_x_bound], [y_pos, y_pos], '--', 
                         color=color, alpha=0.5, linewidth=1.5)
    
    # Add coordinate grid
    ax.grid(True)
    
    # Add axis labels and title
    ax.set_xlabel('x (μm)')
    ax.set_ylabel('y (μm)')
    ax.set_title(f'Layout of {main_cell.name} cell in GDS file - Six-Waveguide CNOT Gate')
    
    # Identify the CNOT coupling region (approximate)
    # In a real scenario, this would be more precisely determined from the design
    central_x = (min_x_bound + max_x_bound) / 2
    central_y = (min_y + max_y) / 2
    coupling_width = 4.8  # From the paper (4.8 μm)
    coupling_height = max_y - min_y - 2*padding
    
    # Draw a rectangle around the estimated coupling region
    coupling_rect = Rectangle((central_x - coupling_width/2, min_y + padding/2), 
                             coupling_width, coupling_height - padding,
                             linewidth=2, edgecolor='g', facecolor='none', 
                             linestyle='--', alpha=0.7)
    ax.add_patch(coupling_rect)
    ax.text(central_x, max_y - padding/2, "CNOT Coupling Region", 
             ha='center', va='bottom', color='g', fontweight='bold')
    
    # Print the coupling region coordinates for use in simulation
    print(f"\nEstimated Coupling Region:")
    print(f"  Center: x ≈ {central_x:.4f}, y ≈ {central_y:.4f}")
    print(f"  Width: {coupling_width} μm")
    print(f"  Input x: {central_x - coupling_width/2:.4f} μm")
    print(f"  Output x: {central_x + coupling_width/2:.4f} μm")
    
    # Mark source and detection positions for simulation
    src_x = central_x - coupling_width/2 - 2.0
    det_x = central_x + coupling_width/2 + 2.0
    
    # Add vertical lines for source and detector positions
    ax.axvline(x=src_x, ymin=0, ymax=1, color='m', linestyle='--', alpha=0.7)
    ax.axvline(x=det_x, ymin=0, ymax=1, color='c', linestyle='--', alpha=0.7)
    
    # Label the source and detection lines
    ax.text(src_x, max_y - 0.5, "Sources", ha='center', va='top', color='m')
    ax.text(det_x, max_y - 0.5, "Detectors", ha='center', va='top', color='c')
    
    # Print suggested source and detector positions
    print(f"\nSuggested Simulation Positions:")
    print(f"  Source x: {src_x:.4f} μm")
    print(f"  Detector x: {det_x:.4f} μm")
    
    # Create a legend for the waveguides
    legend_elements = []
    for i, label in enumerate(waveguide_labels[:len(prominent_bins[:6])]):
        color = waveguide_colors[i % len(waveguide_colors)]
        legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, label=f"WG{i+1}: {label}"))
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    # Save the figure
    plt.tight_layout()
    plt.savefig("gds_visualization_detailed.png", dpi=300)
    print("\nDetailed visualization saved as 'gds_visualization_detailed.png'")
    
else:
    print("No cells found in the GDS file.") 