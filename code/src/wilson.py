"""
TODO 
- wilson loops 
- wilson lines ⊐, ⊔ (missing edge is argument) 

should try to use flax so that it is differentiable
"""

import flax.linen as nn 
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

def wilson_loop(i, x, graph): 
    """
    There is an unique Wilson loop for each lattice site. `i` indexes site for 
    this Wilson loop.
    """
    L = int(np.sqrt(graph.n_nodes))
    left = x[i] 
    right = x[i + 1]
    upper = x[i + L**2]
    lower = x[i + L**2 + L]


    pass


def wilson_loops(x, graph): 
    pass

def wilson_lines(x, graph): 
    pass 