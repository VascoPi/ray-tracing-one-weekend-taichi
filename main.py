import taichi as ti
from vector import Point, vector
from ray import Ray
from hittable import HittableList
from sphere import Sphere

ti.init(arch=ti.gpu)


# Image
image_width = 256
aspect_ratio = 16/9
image_height = int(image_width / aspect_ratio)
pixels = ti.Vector.field(n=3, dtype=ti.f32, shape=(image_width, image_height))


# Camera
viewport_height = 2.0
viewport_width = viewport_height * aspect_ratio
focal_length = 1.0

origin = Point(0.0, 0.0, 0.0)
horizontal = vector(viewport_width, 0.0, 0.0)
vertical = vector(0.0, viewport_height, 0.0)
lower_left_corner = Point(origin - horizontal / 2 - vertical / 2 - vector(0.0, 0.0, focal_length))

# World
world = HittableList()
world.objects.append(Sphere(center=Point(0.0, 0.0, -1.0), radius=0.5))
world.objects.append(Sphere(center=Point(0.0, -100.5, -1.0), radius=100.0))


@ti.kernel
def paint():
    for i, j in pixels:
        u = i / (image_width - 1)
        v = j / (image_height - 1)
        direction = lower_left_corner + u * horizontal + v * vertical - origin
        ray = Ray(origin, direction)
        pixels[i, j] = ray.ray_color(world)


gui = ti.GUI(name='Render', res=(image_width, image_height), show_gui=True)


paint()

while gui.running:
    gui.set_image(pixels)
    gui.show()
gui.set_image(pixels)
gui.show("output.png")

