import math

import open3d as o3d
import numpy as np


def pc_numpy_2_o3d(xyz):
    pcd_bg = o3d.geometry.PointCloud()
    pcd_bg.points = o3d.utility.Vector3dVector(xyz)
    return pcd_bg


centerCamPoint = np.asarray([0, 0, 0.3])

"""
checkInclusionBasedOnTriangleMesh

Creates a mask the size of the points array
True is included in the mesh
False is not included in the mesh
"""


def checkInclusionBasedOnTriangleMesh(pc, mesh):
    points = np.asarray(pc.points)
    legacyMesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(legacyMesh)
    query_point = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
    occupancy = scene.compute_occupancy(query_point).numpy()
    mask = (occupancy == 1)
    return mask


def getLidarShadowMesh(mesh):
    hull, _ = mesh.compute_convex_hull()
    hullVertices = np.asarray(hull.vertices)

    castHullPoints = np.array([])
    for point1 in hullVertices:

        ba = centerCamPoint - point1
        baLen = math.sqrt((ba[0] * ba[0]) + (ba[1] * ba[1]) + (ba[2] * ba[2]))
        ba2 = ba / baLen
        pt2 = centerCamPoint + ((-100) * ba2)

        if (np.size(castHullPoints)):
            castHullPoints = np.vstack((castHullPoints, [pt2]))
        else:
            castHullPoints = np.array([pt2])

    pcdCastHull = o3d.geometry.PointCloud()
    pcdCastHull.points = o3d.utility.Vector3dVector(castHullPoints)
    hull2, _ = pcdCastHull.compute_convex_hull()
    hull2Vertices = np.asarray(hull2.vertices)
    combinedVertices = np.vstack((hullVertices, hull2Vertices))
    pcdShadow = o3d.geometry.PointCloud()
    pcdShadow.points = o3d.utility.Vector3dVector(combinedVertices)
    shadowMesh, _ = pcdShadow.compute_convex_hull()
    return shadowMesh


def get_delete_points_idx(obj, bg):
    # if isinstance(obj, np.ndarray):
    #     obj_pcd = pc_numpy_2_o3d(obj)
    #     mesh = obj_pcd.compute_convex_hull()
    # elif isinstance(obj, o3d.geometry.PointCloud):
    #     mesh = obj.compute_convex_hull()
    # elif isinstance(obj, o3d.geometry.TriangleMesh):
    #     mesh = obj
    # else:
    #     raise ValueError()
    bg = pc_numpy_2_o3d(bg)
    obj_shadow_mesh = getLidarShadowMesh(obj)
    occ_delete_mask = checkInclusionBasedOnTriangleMesh(bg, obj_shadow_mesh)
    return occ_delete_mask
