import taichi as ti
from vector import *
from hittable import HitRecord


@ti.dataclass
class Ray:
    origin: Point
    direction: vector
        
    @ti.func
    def at(self, t):
        return self.origin + self.direction * t

    @ti.func
    def ray_color(self, world):
        color = Color(1.0, 0.0, 0.0)
        hit_record = HitRecord()
        if world.hit(self, 0.0, 99999.0, hit_record):
            color = Color(hit_record.normal + 1.0) * 0.5
        else:
            unit_direction = self.direction.normalized()
            t = 0.5 * (unit_direction.y + 1.0)
            color = 1.0 + t * (Color(0.5, 0.7, 1.0) - 1)
        return color
