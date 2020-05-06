import cv2
import numpy as np

def find_edge_images_correspondences(edges_mesh, edges_target):
    # First find some 'landmarks' in the mesh
    # For now just take the one with the biggest 'y' value
    mesh_non_zeros = cv2.findNonZero(edges_mesh)
    #landmarks = [mesh_non_zeros[0][0]]
    mesh_landmark = mesh_non_zeros[0][0]

    """
    # now find the closest points to it on edges_target
    nonzero_target = cv2.findNonZero(edges_target)
    distances = np.sqrt((nonzero_target[:,:,0] - landmark[0]) ** 2 + (nonzero_target[:,:,1] - landmark[1]) ** 2)
    nearest_index = np.argmin(distances)
    #return [landmark, nonzero_target[nearest_index]]
    return [landmark, nonzero_target[nearest_index][0]]
    """
    # For now just take the one with the biggest 'y' value
    mesh_non_zeros = cv2.findNonZero(edges_target)
    #landmarks = [mesh_non_zeros[0][0]]
    target_landmark = mesh_non_zeros[0][0]
    return [mesh_landmark, target_landmark]