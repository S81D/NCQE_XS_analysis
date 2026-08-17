# needed functions to calculate reconstructed parameters using in selection cuts

import numpy as np
from .constants import bunch_sigma, bunch_offset_mapping

def reco_energy(pe):
    return (pe + 2.90)/5.90

def bunch_cutting(cluster_time, centers):
    avg_diff = bunch_offset_mapping
    return np.any(np.abs( (centers - avg_diff) - cluster_time) < bunch_sigma)

def pairwise_relative_direction(hitX, hitY, hitZ, hitPE, hitT):
    
    hitX = np.array(hitX)
    hitY = np.array(hitY)
    hitZ = np.array(hitZ)
    hitPE = np.array(hitPE)
    hitT = np.array(hitT)

    # Combine and sort by time
    hits = list(zip(hitX, hitY, hitZ, hitPE, hitT))
    hits.sort(key=lambda h: h[4])  # sort by time
    
    #filtered_hits = hits[:4]
    
    mini = min(hitT)
    filtered_hits = []
    # Process remaining hits
    for hit in hits:
        x_i, y_i, z_i, pe_i, t_i = hit

        if (t_i - mini) > 13:
            continue
            
        filtered_hits.append(hit)

    # Extract filtered values
    filtX = np.array([h[0] for h in filtered_hits])
    filtY = np.array([h[1] for h in filtered_hits])
    filtZ = np.array([h[2] for h in filtered_hits])
    filtPE = np.array([h[3] for h in filtered_hits])
    
    n = len(filtX)
    if n < 2:
        return np.array([0.0, 0.0, 0.0])
    
    vec = np.zeros(3)
    for i in range(n):
        for j in range(i+1, n):
            dx = filtX[j] - filtX[i]
            dy = filtY[j] - filtY[i]
            dz = filtZ[j] - filtZ[i]
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            if r == 0:
                continue
            
            # Direction between hits
            dvec = np.array([dx, dy, dz]) / r
            weight = filtPE[i] * filtPE[j]
            vec += weight * dvec
    
    norm = np.linalg.norm(vec)
    if norm == 0:
        return np.array([0.0, 0.0, 0.0])
    return vec / norm

def is_inside_rotated_parabola(Z_val, Y_val, a=-0.45, c=0.4, b=0.0, theta_deg=275):
    # Convert degrees to radians
    theta = np.deg2rad(theta_deg)
    
    # Inverse rotation (world → local)
    u =  Z_val*np.cos(theta) + Y_val*np.sin(theta)
    v = -Z_val*np.sin(theta) + Y_val*np.cos(theta)
    
    # Equation in local coords: v = a*u^2 + b*u + c
    # If opening is along +v, "inside" means v >= parabola
    return v < a*u**2 + b*u + c
