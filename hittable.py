import taichi as ti
from vector import Point, vector, Color


@ti.data_oriented
class HittableList:
    def __init__(self):
        self.objects = []

    @ti.func
    def hit(self, ray, t_min, t_max, hit_record: ti.template()):
        temp_hit_record = HitRecord()
        hit_anything = False
        closest_so_far = t_max
        for i in ti.static(range(len(self.objects))):
            if self.objects[i].hit(ray, t_min, closest_so_far, temp_hit_record):
                hit_anything = True
                closest_so_far = temp_hit_record.t
                hit_record.point = temp_hit_record.point
                hit_record.normal = temp_hit_record.normal
                hit_record.t = temp_hit_record.t
                hit_record.front_face = temp_hit_record.front_face

        return hit_anything


@ti.dataclass
class HitRecord:
    point: Point
    normal: Color
    t: ti.f32
    front_face: ti.i32


@ti.func
def set_face_normal(ray, outward_normal, hit_record: ti.template()):
    hit_record.front_face = ray.direction.dot(outward_normal) < 0
    hit_record.normal = outward_normal if hit_record.front_face == 1 else -outward_normal
