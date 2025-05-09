import numpy as np
import matplotlib.pyplot as plt

def visualize_grid():
    # Load the saved numpy files
    try:
        binary_grid = np.load("../occupancy_grid_viz/binary_occupancy_grid.npy")
        dilated_grid = np.load("../occupancy_grid_viz/dilated_occupancy_grid.npy")
    except FileNotFoundError:
        print("Error: Could not find the numpy files. Make sure they exist in the correct location.")
        return

    # Create figure with grid
    plt.figure(figsize=(15, 15))
    
    # Plot binary grid
    plt.imshow(binary_grid, cmap='gray', origin='lower')
    
    # Plot dilated grid with transparency
    masked_dilated_grid = np.ma.masked_where(dilated_grid == 0, dilated_grid)
    plt.imshow(masked_dilated_grid, cmap='Blues', origin='lower', alpha=0.5)
    
    # Add grid lines
    plt.grid(True, which='both', color='gray', linestyle='-', alpha=0.3)
    
    # Add coordinate labels
    plt.xticks(np.arange(0, binary_grid.shape[1], 50))
    plt.yticks(np.arange(0, binary_grid.shape[0], 50))
    
    # Add title and labels
    plt.title("Occupancy Grid with Dilation and Shape Masks\nClick to see coordinates", fontsize=12)
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    
    # Add colorbar
    plt.colorbar(label='Occupancy')
    
    # Add text showing image dimensions
    plt.figtext(0.02, 0.02, f'Image dimensions: {binary_grid.shape}', fontsize=10)

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            print(f'Pixel coordinates: x={x}, y={y}')
            # Note: To get map coordinates, you would need the transformation parameters
            # from the original map message. For now, we'll just show pixel coordinates.

    # Connect the click event
    plt.connect('button_press_event', onclick)
    
    # Show the interactive plot
    plt.show()

if __name__ == "__main__":
    visualize_grid()