import taichi as ti
from vector import Point, Color
from material import MaterialData


@ti.data_oriented
class HittableList:
    def __init__(self):
        self.objects = []

    @ti.func
    def hit(self, ray, t_min, t_max):
        material_data = MaterialData()
        hit_record = HitRecord()
        hit_anything = False
        closest_so_far = t_max
        for i in ti.static(range(len(self.objects))):
            temp_hit_record, is_hit = self.objects[i].hit(ray, t_min, closest_so_far)
            if is_hit:
                hit_anything = True
                closest_so_far = temp_hit_record.t
                material_data = self.objects[i].material.material_data
                hit_record = temp_hit_record

        return hit_record, hit_anything, material_data


@ti.dataclass
class HitRecord:
    point: Point
    normal: Color
    t: ti.f32
    material_data: MaterialData
    front_face: ti.i32


@ti.func
def set_face_normal(ray, outward_normal):
    front_face = ray.direction.dot(outward_normal) < 0
    normal = outward_normal if front_face else -outward_normal

    return front_face, normal
