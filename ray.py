import taichi as ti
from vector import Vector, Point


@ti.dataclass
class Ray:
    origin: Point
    direction: Vector
        
    @ti.func
    def at(self, t):
        return self.origin + self.direction * t
