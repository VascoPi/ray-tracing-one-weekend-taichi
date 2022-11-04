import taichi as ti
from vector import Point, Color
from material import MaterialData, Dielectric, Metal, Lambert
import random


def random_scene():
    world = HittableList()

    material1 = Dielectric(1.5)
    material2 = Lambert(Color(0.4, 0.2, 0.1))
    material3 = Metal(Color(0.7, 0.6, 0.5), 0.0)

    world.objects.append(Sphere(center=Point(0.0, 1.0, 0.0), radius=1.0, material=material1))
    world.objects.append(Sphere(center=Point(-4.0, 1.0, 0.0), radius=1.0, material=material2))
    world.objects.append(Sphere(center=Point(4.0, 1.0, 0.0), radius=1.0, material=material3))

    for a in range(-11, 11):
        for b in range(-1, 1):
            material_type = random.random()
            center = Point(a + 0.9*random.random())
            if center - Point(4.0, 0.2, 0.0).norm() > 0.9:
                if material_type < 0.8:
                    material = Lambert(Color(random.random(), random.random(), random.random()))
                elif material_type < 0.95:
                    material = Metal(Color(random.random(), random.random(), random.random()) * 0.5 + 0.5, random.random() * 0.5)
                else:
                    material = Dielectric(1.5)

            world.objects.append(Sphere(center=center, radius=0.2, material=material))

    material_ground = Lambert(Color(0.5, 0.5, 0.5))
    world.objects.append(Sphere(center=Point(0.0, -1000.0, 0.0), radius=1000.0, material=material_ground))

    return world


@ti.data_oriented
class HittableList:
    def __init__(self):
        self.objects = []

    @ti.func
    def hit(self, ray, t_min, t_max):
        mat_info = MaterialData()
        hit_record = HitRecord()
        hit_anything = False
        closest_so_far = t_max
        for i in ti.static(range(len(self.objects))):
            is_hit, temp_hit_record = self.objects[i].hit(self.objects[i].get_info()[0], ray, t_min, closest_so_far)
            if is_hit:
                hit_anything = True
                closest_so_far = temp_hit_record.t
                mat_info = self.objects[i].material.mat_info
                hit_record = temp_hit_record

        return hit_anything, hit_record, mat_info


SphereInfo = ti.types.struct(center=Point, radius=ti.f32)


@ti.data_oriented
class Sphere:
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material

    def get_info(self):
        return SphereInfo(center=self.center, radius=self.radius), self.material.mat_info

    @staticmethod
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
                set_face_normal(ray, outward_normal, hit_record)

        return result, hit_record


@ti.dataclass
class HitRecord:
    point: Point
    normal: Color
    t: ti.f32
    mat_info: MaterialData
    front_face: ti.i32


@ti.func
def set_face_normal(ray, outward_normal, hit_record: ti.template()):
    hit_record.front_face = ray.direction.dot(outward_normal) < 0
    hit_record.normal = outward_normal if hit_record.front_face else -outward_normal
