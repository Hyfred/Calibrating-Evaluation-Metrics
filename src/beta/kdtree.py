import collections
import operator

import numpy as np
from sklearn.decomposition import PCA

BT = collections.namedtuple("BT", ["value", "left", "right"])


class KDTree(object):

    def __init__(self, max_depth, min_points=15, use_low_dim_projection=False):
        self.max_depth = max_depth
        self.bt = None
        self.min_points = min_points
        self.use_low_dim_projection = use_low_dim_projection
        
        if use_low_dim_projection:
            self.pca = PCA(n_components=max_depth)
            
        self.boundaries = {
            "min": {},
            "max": {},
        }

    def fit(self, points):
        
        if self.use_low_dim_projection:
            points = self.pca.fit_transform(points)

        k = len(points[0])

        def build(*, points, depth):
            """Build a k-d tree from a set of points at a given
            depth.
            """
            if len(points) < self.min_points:
                return None

            if len(points) == 0:
                return None

            if depth == self.max_depth:
                return None

            points.sort(key=operator.itemgetter(depth % k))
            middle = len(points) // 2

            min_coord_at_depth = np.min(np.array(points)[:, depth % k])
            if depth in self.boundaries["min"]:
                self.boundaries["min"][depth] = min(
                    self.boundaries["min"][depth], min_coord_at_depth
                )
            else:
                self.boundaries["min"][depth] = min_coord_at_depth

            max_coord_at_depth = np.max(np.array(points)[:, depth % k])
            if depth in self.boundaries["max"]:
                self.boundaries["max"][depth] = max(
                    self.boundaries["max"][depth], max_coord_at_depth
                )
            else:
                self.boundaries["max"][depth] = max_coord_at_depth

            return BT(
                value=points[middle],
                left=build(
                    points=points[:middle],
                    depth=depth + 1,
                ),
                right=build(
                    points=points[middle + 1 :],
                    depth=depth + 1,
                ),
            )

        self.bt = build(points=list(points), depth=0)

        self.boundaries["min"] = np.array(list(self.boundaries["min"].values()))
        self.boundaries["max"] = np.array(list(self.boundaries["max"].values()))

    def get_partition_indices(self, points):

        if self.use_low_dim_projection:
            points = self.pca.fit_transform(points)

        def _is_within_boundary(point):
            if len(self.boundaries["min"]) == 0 or len(self.boundaries["max"]) == 0:
                return True
            return np.all(
                self.boundaries["min"] <= point[: len(self.boundaries["min"])]
            ) & np.all(self.boundaries["max"] >= point[: len(self.boundaries["max"])])

        def _partition_index(point):
            path_to_leaf = ""

            cur_tree = self.bt
            for idx, elem in enumerate(point):
                if cur_tree is None:
                    break

                if elem <= cur_tree.value[idx]:
                    path_to_leaf += "0"
                    cur_tree = cur_tree.left
                else:
                    path_to_leaf += "1"
                    cur_tree = cur_tree.right

            if path_to_leaf == "":
                return 0
            else:
                return int(path_to_leaf, 2)

        return np.array(
            [
                _partition_index(point) if _is_within_boundary(point) else np.nan
                for point in points
            ]
        )
