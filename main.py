import taichi as ti
from vector import Point, vector
from hittable import HittableList
from sphere import Sphere
from camera import Camera

ti.init(arch=ti.gpu)


# Image
image_width = 256
aspect_ratio = 16/9
image_height = int(image_width / aspect_ratio)
pixels = ti.Vector.field(n=3, dtype=ti.f32, shape=(image_width, image_height))
samples_per_pixel = 100

# World
world = HittableList()
world.objects.append(Sphere(center=Point(0.0, 0.0, -1.0), radius=0.5))
world.objects.append(Sphere(center=Point(0.0, -100.5, -1.0), radius=100.0))

# Camera
camera = Camera(aspect_ratio)


@ti.kernel
def render():
    for i, j in pixels:
        u = (i + ti.random()) / (image_width - 1)
        v = (j + ti.random()) / (image_height - 1)
        ray = camera.get_ray(u, v)
        pixels[i, j] += ray.ray_color(world)


gui = ti.GUI(name='Render', res=(image_width, image_height), show_gui=True)


for k in range(samples_per_pixel):
    render()

gui.set_image(pixels.to_numpy() / samples_per_pixel)
gui.show("output.png")

