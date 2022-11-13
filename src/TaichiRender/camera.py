import taichi as ti
from .vector import Point, random_in_unit_disk
from .ray import Ray
import math


@ti.data_oriented
class Camera:
    def __init__(self, look_from, look_at, vup, vfov, aspect_ratio, aperture, focus_distance):
        self.vfov = vfov

        theta = math.radians(vfov)
        h = math.tan(theta / 2)
        viewport_height = 2.0 * h
        viewport_width = aspect_ratio * viewport_height

        self.focal_length = 1.0

        self.look_from = look_from
        self.look_at = look_at
        self.vup = vup
        self.aperture = aperture
        self.focus_distance = focus_distance
        self.lens_radius = self.aperture / 2

        self.origin = self.look_from
        w = (self.look_from - self.look_at).normalized()
        self.u = vup.cross(w).normalized()
        self.v = w.cross(self.u)

        self.horizontal = self.focus_distance * viewport_width * self.u
        self.vertical = self.focus_distance * viewport_height * self.v
        self.lower_left_corner = Point(
            self.origin - self.horizontal / 2 - self.vertical / 2 - self.focus_distance * w
        )

    @ti.func
    def get_ray(self, s, t):
        rd = self.lens_radius * random_in_unit_disk()
        offset = self.u * rd.x + self.v * rd.y
        return Ray(
            self.origin + offset,
            self.lower_left_corner + s * self.horizontal + t * self.vertical - self.origin - offset,
        )
