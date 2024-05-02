def wrap_index(N, m, n):
    if m >= n:
        swap = m
        m = n
        n = swap
    
    forward_dist = abs(m-n)
    back_dist = abs(N+m-n)
    return min(forward_dist, back_dist)