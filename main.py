from taichi.lang.ops import random
from hittable import HittableList, Sphere, HitRecord
import taichi as ti
from vector import *
from ray import Ray
from camera import Camera


# First we init taichi.  You can select CPU or GPU, or specify CUDA, Metal, etc
ti.init(arch=ti.gpu)

# Setup image data
ASPECT_RATIO = 16.0 / 9.0
IMAGE_WIDTH = 400
IMAGE_HEIGHT = int(IMAGE_WIDTH / ASPECT_RATIO)
SAMPLES_PER_PIXEL = 100

INFINITY = 99999999.9

# This is our pixel array which needs to be setup for the kernel.
# We specify the type and size of the field with 3 channels for RGB
# I set this up with floating point because it will be nicer in the future.
pixels = ti.Vector.field(n=3, dtype=ti.f32, shape=(IMAGE_WIDTH, IMAGE_HEIGHT))
world = HittableList()
world.add(Sphere(Point(0.0, 0.0, -1.0), 0.5))
world.add(Sphere(Point(0.0, -100.5, -1), 100.0))
cam = Camera(ASPECT_RATIO)

# A Taichi function that returns a color gradient of the background based on
# the ray direction.
@ti.func
def ray_color(r, world):
    color = Color(0.0)  # Taichi functions can only have one return call
    rec = HitRecord(p=Point(0.0), normal=Vector(0.0), t=0.0, front_face=1)
    if world.hit(r, 0.0, INFINITY, rec):
        color = 0.5 * (rec.normal + 1.0)
    else:
        unit_direction = r.dir.normalized()
        t = 0.5 * (unit_direction.y + 1.0)
        color = (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0)
    return color


# Our "kernel".  This loops over all the samples for a pixel in a parallel manner
# We don't multiply by 256 as in the original code since we use floats
@ti.kernel
def render_pass():
    for i, j in pixels:
        u, v = (i + ti.random()) / (IMAGE_WIDTH - 1), (j + ti.random()) / (IMAGE_HEIGHT - 1)
        ray = cam.get_ray(u, v)
        pixels[i, j] += ray_color(ray, world)


if __name__ == '__main__':
    gui = ti.GUI("Ray Tracing in One Weekend", res=(IMAGE_WIDTH, IMAGE_HEIGHT))

    # Run the kernel once for each sample
    for i in range(SAMPLES_PER_PIXEL):
        render_pass()

        gui.set_image(pixels.to_numpy() / (i + 1))
        gui.show()  # show in GUI
        print("\rPercent Complete\t:{:.2%}".format((i + 1)/SAMPLES_PER_PIXEL), end='')

    gui.set_image(pixels.to_numpy() / SAMPLES_PER_PIXEL)
    gui.show('out.png')
