import taichi as ti
from vector import Point
from hittable import HittableList
from sphere import Sphere
from camera import Camera
from material import Metal, Lambert, Dielectric, scatter
from vector import Color, Vector

ti.init(arch=ti.gpu)


# Image
image_width = 256
aspect_ratio = 16/9
image_height = int(image_width / aspect_ratio)
pixels = ti.Vector.field(n=3, dtype=ti.f32, shape=(image_width, image_height))
samples_per_pixel = 100

material_ground = Lambert(Color(0.8, 0.8, 0.0))
material_center = Lambert(Color(0.1, 0.2, 0.5))
material_left = Dielectric(1.5)
material_right = Metal(Color(0.8, 0.6, 0.2), 0.0)


# World
world = HittableList()
world.objects.append(Sphere(name="Sphere01", center=Point(0.0, 0.0, -1.0), radius=0.5, material=material_center))
world.objects.append(Sphere(name="Sphere02", center=Point(-1.0, 0.0, -1.0), radius=0.5, material=material_left))
world.objects.append(Sphere(name="Sphere03", center=Point(1.0, 0.0, -1.0), radius=0.5, material=material_right))
world.objects.append(Sphere(name="Sphere04", center=Point(0.0, -100.5, -1.0), radius=100.0, material=material_ground))


# Camera
look_from = Point(3.0, 3.0, 2.0)
look_at = Point(0.0, 0.0, -1.0)
focus_distance = (look_from - look_at).norm()
camera = Camera(look_from, look_at, Vector(0, 1, 0), 20, aspect_ratio, 2.0, focus_distance)


@ti.func
def ray_color(ray, world):
    color = Color(1.0)
    depth = 50

    while depth > 0:
        hit_record, hit, material_data = world.hit(ray, 0.001, 99999.0)
        if hit:
            scattered, ray, attenuation = scatter(material_data, ray.direction, hit_record)
            if scattered:
                color *= attenuation
                depth -= 1
            else:
                color = Color(0.0)
                break
        else:
            unit_direction = ray.direction.normalized()
            t = 0.5 * (unit_direction.y + 1.0)
            color = color * (1.0 + t * (Color(0.5, 0.7, 1.0) - 1))
            break

    return color


@ti.kernel
def render():
    for i, j in pixels:
        u = (i + ti.random()) / (image_width - 1)
        v = (j + ti.random()) / (image_height - 1)
        ray = camera.get_ray(u, v)
        pixels[i, j] += ray_color(ray, world)


gui = ti.GUI(name='Render', res=(image_width, image_height), show_gui=True)

# render()
for k in range(samples_per_pixel):
    render()

gui.set_image((pixels.to_numpy() / samples_per_pixel) ** 0.5)
gui.show("output.png")
