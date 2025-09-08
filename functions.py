from numba import njit, prange
import numpy as np

@njit(fastmath=True, cache=True, nogil=True)
def is_target_blocked(missile_pos, cloud_center, time):
    cloud_radius = 10.0
    if time > 20 or time < 0:
        return False
    dist_mc = np.sqrt(np.sum((missile_pos - cloud_center) ** 2))
    if dist_mc < cloud_radius:
        return True
    if (missile_pos[2] < cloud_center[2] - cloud_radius) or missile_pos[2] < 0:
        return False
    
    target_center = np.array([0.0, 200.0, 0.0])
    target_radius = 7.0
    targrt_height = 10.0
    vec_mc = cloud_center - missile_pos
    norm_mc = np.sqrt(np.sum(vec_mc ** 2))
    dir_mc = vec_mc / norm_mc        
    solid_angle = np.arcsin(cloud_radius / norm_mc)

    n_alpha = 4
    n_h = 2
    for i in prange(n_alpha):
        alpha = 2 * np.pi * i / n_alpha
        x = target_center[0] + target_radius * np.cos(alpha)
        y = target_center[1] + target_radius * np.sin(alpha)
        for j in prange(n_h):
            z = target_center[2] + j * targrt_height / (n_h - 1)
            target_pos = np.array([x, y, z])
            vec_mt = target_pos - missile_pos
            norm_mt = np.sqrt(np.sum(vec_mt ** 2))
            dir_mt = vec_mt / norm_mt
            cos_theta = np.dot(dir_mt, dir_mc)
            theta = np.arccos(min(1.0, max(-1.0, cos_theta)))
            if theta > solid_angle:
                return False
    return True

@njit(fastmath=True, cache=True, nogil=True)
def v_of_fy(speed, theta):
    x = np.cos(theta)
    y = np.sin(theta)
    return speed * np.array([x, y, 0.0])

@njit(fastmath=True, cache=True, nogil=True)
def fy_bomb_position(drone_start_pos, drone_height, theta, speed, t_throw, t_delay, g=np.array([0,0,-9.8])):
    v0 = speed * np.array([np.cos(theta), np.sin(theta), 0.0])
    t_total = t_throw + t_delay
    pos = drone_start_pos + drone_height + v0 * t_total + 0.5 * g * t_delay**2
    return pos

@njit(fastmath=True, cache=True, nogil=True)
def fy_throw_position(drone_start_pos, drone_height, theta, speed, t_throw):
    v0 = speed * np.array([np.cos(theta), np.sin(theta), 0.0])    
    pos = drone_start_pos + drone_height + v0 * t_throw
    return pos
