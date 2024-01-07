import numpy as np

from linear_interpolation import optimize_nodes, compute_errors
from linear_interpolation import plot_graph, plot_3d_graph, plot_contour, plot_constatns_graph_const_N, plot_constatns_graph_const_n
from linear_interpolation import T, C


def lab_advanced(method: str = 'linear'):
    N, n = 10, 10
    A_initial, B_initial = 0, 1
    t_values = np.linspace(0, T, num=100)

    if method == 'linear':
        x_nodes_new, optimized_y_nodes_new_efficient = optimize_nodes(
            N, A_initial, B_initial, n)
        plot_graph(x_nodes_new, optimized_y_nodes_new_efficient, t_values, C, T)

        N_range = n_range = np.arange(3, 31)
        A_initial, B_initial = 0, 1

        h_interp_array, h_integr_array, error_matrix = compute_errors(
            N_range, n_range, A_initial, B_initial)

        plot_3d_graph(h_interp_array, h_integr_array, error_matrix)
        plot_3d_graph(h_interp_array, h_integr_array, error_matrix, elev=45)
        plot_contour(h_interp_array, h_integr_array, error_matrix)
        plot_constatns_graph_const_N(20, 20)
        plot_constatns_graph_const_n(20, 20)
