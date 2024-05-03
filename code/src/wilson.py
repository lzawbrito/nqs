"""
TODO 
- wilson loops 
- wilson lines ⊐, ⊔ (missing edge is argument) 

should try to use flax so that it is differentiable
"""

import jax.numpy as np

def ordered_edges(graph): 
    """
    Assumes square LxL lattice.
    """
    a = [] 
    L = np.sqrt(graph.n_nodes)
    for x in graph.nodes(): 
        a.append((x, int((x + L) % graph.n_nodes)))
    
    for x in graph.nodes(): 
        a.append((x, int(L * (x // L) + (x + 1) % L)))
    
    return a

def plaquette_indices(i, L): 
    """
    Outputs: 
    - L, R, U, D, indices
    """
    #left, right, upper, lower
    return i, int(L * (i // L) + (i + 1) % L), i + L**2, (i + L) % (L**2) + L**2 


def get_wilson_loops_and_lines(x, L):
    """
    Given a list of edges, returns the wilson loops and paths for every vertex in the original lattice.
    Assumes a square lattice of size (L x L) with periodic boundary conditions, the input should be the output of ordered_edges(nk.graph.Square(L, pbc=True))
    Inputs:
    - edges: an ordered list of edges, where each edge is a tuple of the form (x,y) representing the connection between two nodes
    - L: the dimension of the lattice 
    Outputs:
    - wilson_loops: the wilson loops for each point in the graph, which is left_edge * right_edge * upper_edge * lower_edge
    - wilson_paths_left: the wilson paths for each point in the graph, when starting by going left, which is left_edge * lower_edge * right_edge
    - wilson_paths_up: the wilson paths for each point in the graph, when starting by going up, which is upper_edge * right_edge * lower_edge
    """
    wilson_loops      = []
    wilson_paths_left = []
    wilson_paths_up   = []
    for i in range(L**2):
        l, r, u, d = plaquette_indices(i, L)
        left, right, upper, lower = x[:, l], x[:, r], x[:, u], x[:, d]

        wilson_loops.append(left * right * lower * upper) # feels like there should be more efficient way instead of appending
        wilson_paths_up.append(left * lower * right)
        wilson_paths_left.append(upper * right * upper)

    return np.array(wilson_loops).T, np.array(wilson_paths_left).T, np.array(wilson_paths_up).T

