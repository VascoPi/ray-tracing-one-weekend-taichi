import taichi as ti
from hittable import set_face_normal, HitRecord


@ti.data_oriented
class Sphere:
    def __init__(self, name, center, radius, material):
        self.name = name
        self.center = center
        self.radius = radius
        self.material = material

    @ti.func
    def hit(self, ray, t_min, t_max):
        result = False
        hit_record = HitRecord()
        oc = ray.origin - self.center
        a = ray.direction.norm_sqr()
        half_b = oc.dot(ray.direction)
        c = oc.norm_sqr() - self.radius ** 2
        discriminant = half_b ** 2 - a * c
        if discriminant >= 0.0:
            sqrtd = ti.sqrt(discriminant)
            root = (-half_b - sqrtd) / a
            if root < t_min or t_max < root:
                root = (-half_b + sqrtd) / a

            result = not (root < t_min or t_max < root)

            if result:
                hit_record.t = root
                hit_record.point = ray.at(hit_record.t)
                outward_normal = (hit_record.point - self.center) / self.radius
                hit_record.front_face, hit_record.normal = set_face_normal(ray, outward_normal)

        return hit_record, result
