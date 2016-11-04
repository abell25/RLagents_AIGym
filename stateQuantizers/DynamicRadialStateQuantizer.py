import numpy as np

class DynamicRadialStateQuanizer:
    """ Generates new states when a new point is more than `grid_radius` farther from the nearest point"""

    def __init__(self, grid_radius):
        self.grid_radius = grid_radius
        self.point_lookup = {}
        self.points = []

    def s_to_discrete(self, s):
        # return tuple([np.sign(n) for n in s])
        return tuple(self.get_s_point(s))

    def get_s_point(self, s):
        s_t = tuple(s)
        if s_t in self.point_lookup:
            return self.point_lookup[s_t]

        if s_t in self.points:
            return s

        if len(self.points) == 0:
            self.points.add(s_t)
            self.point_lookup[s_t] = s_t
            return s

        d = lambda x, y: np.sum((x - y) ** 2)
        closest_p = min(self.points, key=lambda x: d(x, s))
        if d(s, closest_p) >= self.grid_radius:
            self.points.add(s_t)
            self.point_lookup[s_t] = s_t
            return s
        else:
            self.point_lookup[s_t] = closest_p
            return closest_p

