import taichi as ti
from .ray import Ray
from .vector import Color, random_in_unit_sphere, Point, Vector, near_zero

LAMBERT = 0
METAL = 1
DIELECTRIC = 2


@ti.func
def reflect(vector, normal):
    return vector - 2 * vector.dot(normal) * normal


@ti.func
def reflectance(cosine, idx):
    r0 = ((1.0 - idx) / (1.0 + idx)) ** 2
    return r0 + (1.0 - r0) * ((1.0 - cosine) ** 5)


@ti.func
def reflect(v, n):
    return v - 2.0 * v.dot(n) * n


@ti.func
def refract(v, n, etai_over_etat):
    cos_theta = ti.min(-v.dot(n), 1.0)
    r_out_perp = etai_over_etat * (v + cos_theta * n)
    r_out_parallel = -ti.sqrt(abs(1.0 - r_out_perp.norm_sqr())) * n
    return r_out_perp + r_out_parallel


@ti.dataclass
class MaterialData:
    color: Color
    roughness: ti.f32
    ior: ti.f32
    mat_type: ti.i32


@ti.data_oriented
class Metal:
    def __init__(self, color, roughness):
        self.mat_info = MaterialData(color=color, roughness=roughness, ior=1.0, mat_type=METAL)

    @staticmethod
    @ti.func
    def scatter(mat_info, in_direction, hit_record):
        out_direction = (
            reflect(in_direction.normalized(), hit_record.normal)
            + mat_info.roughness * random_in_unit_sphere()
        )
        reflected = out_direction.dot(hit_record.normal) > 0.0
        return reflected, Ray(origin=hit_record.point, direction=out_direction), mat_info.color


@ti.data_oriented
class Lambert:
    def __init__(self, color):
        self.mat_info = MaterialData(color=color, roughness=0.0, ior=1.0, mat_type=LAMBERT)

    @staticmethod
    @ti.func
    def scatter(mat_info, in_direction, rec):
        out_direction = rec.normal + random_in_unit_sphere()

        if near_zero(out_direction):
            vec = rec.normal

        return True, Ray(origin=rec.point, direction=out_direction), mat_info.color


@ti.data_oriented
class Dielectric:
    def __init__(self, ior):
        self.mat_info = MaterialData(color=Color(1.0), roughness=0.0, ior=ior, mat_type=DIELECTRIC)

    @staticmethod
    @ti.func
    def scatter(mat_info, in_direction, rec):
        refraction_ratio = 1.0 / mat_info.ior if rec.front_face else mat_info.ior
        unit_dir = in_direction.normalized()
        cos_theta = ti.min(-unit_dir.dot(rec.normal), 1.0)
        sin_theta = ti.sqrt(1.0 - cos_theta * cos_theta)

        out_direction = Vector(0.0, 0.0, 0.0)
        cannot_refract = refraction_ratio * sin_theta > 1.0
        if cannot_refract or reflectance(cos_theta, refraction_ratio) > ti.random():
            out_direction = reflect(unit_dir, rec.normal)
        else:
            out_direction = refract(unit_dir, rec.normal, refraction_ratio)

        return True, Ray(origin=rec.point, direction=out_direction), mat_info.color


@ti.func
def scatter(mat_info, in_direction, hit_record):
    hit = False
    scattered = Ray(origin=Point(0.0), direction=Vector(0.0))
    attenuation = Color(0.0, 0.0, 0.0)
    if mat_info.mat_type == LAMBERT:
        hit, scattered, attenuation = Lambert.scatter(mat_info, in_direction, hit_record)
    elif mat_info.mat_type == METAL:
        hit, scattered, attenuation = Metal.scatter(mat_info, in_direction, hit_record)
    else:
        hit, scattered, attenuation = Dielectric.scatter(mat_info, in_direction, hit_record)

    return hit, scattered, attenuation
