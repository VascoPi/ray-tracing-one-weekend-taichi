import taichi as ti
import taichi_glsl


Vector = ti.types.vector(3, ti.f32)
Color = Vector
Point = Vector


@ti.func
def random_in_unit_sphere():
    return taichi_glsl.randgen.randUnit3D()


@ti.func
def random_in_hemi_sphere(normal):
    vec = taichi_glsl.randgen.randUnit3D()
    if vec.dot(normal) <= 0.0:
        vec = -vec

    return vec


@ti.func
def random_in_unit_disk():
    vec = taichi_glsl.randgen.randUnit3D()
    return Vector(vec[0], vec[1], 0.0)


@ti.func
def near_zero(vector):
    s = 1e-8
    return abs(vector[0]) < s and abs(vector[1]) < s and abs(vector[2]) < s
