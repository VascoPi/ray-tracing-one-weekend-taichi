import taichi as ti
from vector import vector, Point
from ray import Ray

@ti.data_oriented
class Camera:
    def __init__(self, aspect_ratio):
        self.viewport_height = 2.0
        self.viewport_width = aspect_ratio * self.viewport_height

        self.focal_length = 1.0

        self.origin = Point(0.0, 0.0, 0.0)
        self.horizontal = vector(self.viewport_width, 0.0, 0.0)
        self.vertical = vector(0.0, self.viewport_height, 0.0)
        self.lower_left_corner = Point(self.origin - self.horizontal / 2 - self.vertical / 2 - vector(0.0, 0.0, self.focal_length))

    @ti.func
    def get_ray(self, u, v):
        return Ray(self.origin, self.lower_left_corner + u * self.horizontal + v * self.vertical - self.origin)