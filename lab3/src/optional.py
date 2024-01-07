import sys
from matplotlib.ticker import MultipleLocator, FuncFormatter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Callable, Tuple, List
from main import f, runge_kutta


def classify_trajectory(y_values: np.ndarray, threshold: float = 0.01) -> float:
    # Consider the mean of the last few positions as the asymptotic state
    return np.floor(np.mean(y_values[-50:, 0])/np.pi)


def create_basins_of_attraction(
    f: Callable[[float, np.ndarray], np.ndarray],
    runge_kutta: Callable[[np.ndarray, float, Callable, float], Tuple[np.ndarray, np.ndarray]],
    theta_range: Tuple[float, float],
    theta_dot_range: Tuple[float, float],
    grid_size_theta: int,
    grid_size_theta_dot: int,
    t_final: float,
    h: float
) -> List[Tuple[float, float, float]]:
    # Create a grid of initial conditions
    theta_values = np.linspace(*theta_range, grid_size_theta)
    theta_dot_values = np.linspace(*theta_dot_range, grid_size_theta_dot)
    initial_conditions = [(theta0, theta_dot0)
                          for theta0 in theta_values for theta_dot0 in theta_dot_values]

    # Initialize the basin classification array
    basin_classifications = []

    # Calculate the total number of iterations
    total_iterations = len(initial_conditions)

    # Calculate the trajectory and classify the asymptotic state for each initial condition
    for i, (theta0, theta_dot0) in enumerate(initial_conditions):
        # Calculate the trajectory using the provided Runge-Kutta method
        _, trajectory = runge_kutta(
            np.array([theta0, theta_dot0]), t_final, f, h)
        # Classify the asymptotic state of the trajectory
        classification = classify_trajectory(trajectory)
        basin_classifications.append((theta0, theta_dot0, classification))

        # Calculate the progress
        progress = 100.0 * (i + 1) / total_iterations

        # Print the progress
        sys.stdout.write("\rProgress: %.2f%%" % progress)
        sys.stdout.flush()

    return basin_classifications


# Define parameters
theta_range = (-4 * np.pi, 4 * np.pi)
theta_dot_range = (-5, 5)

grid_size_theta = 300
grid_size_theta_dot = 200
t_final = 300

h = 0.1

# Get the basin classifications with the new grid size
basin_classifications = create_basins_of_attraction(
    f, runge_kutta, theta_range, theta_dot_range, grid_size_theta, grid_size_theta_dot, t_final, h)

# Create a colormap
cmap = plt.colormaps['magma']

# Normalize the classifications to the range [0, 1] for the colormap
classifications = np.array(
    [classification for _, _, classification in basin_classifications])
norm = plt.Normalize(classifications.min(), classifications.max())

# Plot the basins of attraction using the colormap


# Define grid count and grid lengths
grid_count = (grid_size_theta, grid_size_theta_dot)
grid_x_len = (theta_range[1] - theta_range[0]) / grid_count[0]
grid_y_len = (theta_dot_range[1] - theta_dot_range[0]) / grid_count[1]

# Plot the basins of attraction using the colormap
fig, ax = plt.subplots(figsize=(12, 8))
for theta0, theta_dot0, classification in basin_classifications:
    color = cmap(norm(classification))
    rect = plt.Rectangle((theta0, theta_dot0), grid_x_len,
                         grid_y_len, color=color)
    ax.add_patch(rect)

# Add a colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)

# Set x ticks
ax.xaxis.set_major_locator(MultipleLocator(base=np.pi))
ax.xaxis.set_major_formatter(FuncFormatter(
    lambda val, pos: '{:.0f}$\pi$'.format(val/np.pi) if val != 0 else '0'))

# Set y ticks
# Set interval between y ticks to 1
ax.yaxis.set_major_locator(MultipleLocator(base=1))

ax.set_xlabel(f'$\\theta(t)$')
ax.set_ylabel(f'$d\\theta(t)/dt$')
# ax.set_title('Basins of Attraction')
ax.set_xlim(theta_range)
ax.set_ylim(theta_dot_range)
plt.tight_layout()
# plt.show()

# Save the figure
plt.savefig('images/Basins-Of-Attraction.pdf', format='pdf')
