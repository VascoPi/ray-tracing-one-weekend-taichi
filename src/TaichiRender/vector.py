import math

import taichi as ti
from taichi.math import vec3
from taichi.math import vec4


Vector = ti.types.vector(3, ti.f32)
Color = Vector
Point = Vector


@ti.func
def near_zero(vector):
    s = 1e-8
    return abs(vector[0]) < s and abs(vector[1]) < s and abs(vector[2]) < s


@ti.func
def random_in_unit_disk():
    """Create a vector to a random point in disk."""
    theta = ti.random() * math.pi * 2.0
    r = ti.random() ** 0.5

    return ti.math.vec3(r * ti.cos(theta), r * ti.sin(theta), 0.0)


@ti.func
def random_in_hemi_sphere(normal):
    """Create a vector to a random point in a Hemisphere."""
    vec = random_in_unit_sphere()
    if vec.dot(normal) < 0:
        vec = -vec
    return vec


@ti.func
def random_in_unit_sphere():
    """Create a vector to a random point in Sphere."""
    theta = ti.random() * math.pi * 2.0
    v = ti.random()
    phi = ti.acos(2.0 * v - 1.0)
    r = ti.random() ** (1 / 3)
    return vec3(
        r * ti.sin(phi) * ti.cos(theta),
        r * ti.sin(phi) * ti.sin(theta),
        r * ti.cos(phi),
    )


@ti.func
def convert_space_point(mat, vec):
    """Convert a point using a matrix."""
    vec_new = mat @ vec4(vec.x, vec.y, vec.z, 1.0)
    return vec3(vec_new.x, vec_new.y, vec_new.z)


@ti.func
def convert_space_vector(mat, vec):
    """Convert a vector using a matrix."""
    vec_new = mat @ vec4(vec.x, vec.y, vec.z, 0.0)
    return vec3(vec_new.x, vec_new.y, vec_new.z)
