from Utils_common import get_geometric_info


class ObjectInsert():
    def __init__(self):
        self.idx = None

        self.mesh = None
        self.barycenter_xy = None
        self.half_diagonal = None

        self.img_obj = None
        self.coordinate = None

        self.pcd_obj = None

        self.position = None
        self.rz_degree = None

        self.box3d = None
