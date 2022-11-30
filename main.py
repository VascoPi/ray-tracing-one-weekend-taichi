import taichi as ti
from vector import Point
from camera import Camera
from material import scatter
from vector import Color, Vector
from hittable import random_scene
import time

ti.init(arch=ti.gpu)

# Image
image_width = 800
aspect_ratio = 3.0 / 2.0
image_height = int(image_width / aspect_ratio)
pixels = ti.Vector.field(n=3, dtype=ti.f32, shape=(image_width, image_height))
samples_per_pixel = 100

# Camera
look_from = Point(13.0, 2.0, 3.0)
look_at = Point(0.0, 0.0, 0.0)
focus_distance = (look_from - look_at).norm()
focus_distance = 10.0
aperture = 0.1
camera = Camera(look_from, look_at, Vector(0, 1, 0), 20, aspect_ratio, aperture, focus_distance)

# World
world = random_scene(is_bvh=True)


@ti.func
def ray_color(ray, world):
    color = Color(1.0)
    depth = 50

    while depth > 0:
        hit, hit_record, mat_info = world.hit(ray, 0.001, 99999.0)
        if hit:
            scattered, out_ray, attenuation = scatter(mat_info, ray.direction, hit_record)
            if scattered:
                color *= attenuation
                ray = out_ray
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
        # if not(i == 1 and j == 1):
        #     continue
        u = (i + ti.random()) / (image_width - 1)
        v = (j + ti.random()) / (image_height - 1)
        ray = camera.get_ray(u, v)
        pixels[i, j] += ray_color(ray, world)


world.commit()
# gui = ti.GUI(name='Render', res=(image_width, image_height))
#
# t = time.time()
# render()
# for k in range(samples_per_pixel):
#     render()
#     gui.set_image((pixels.to_numpy() / k) ** 0.5)
#     gui.show()
# print(time.time() - t)
# gui.set_image((pixels.to_numpy() / samples_per_pixel) ** 0.5)
# gui.show("output.png")


window = ti.ui.Window(name="Render", res=(image_width, image_height))
gui = window.get_gui()
canvas = window.get_canvas()
scene = ti.ui.Scene()

cam = ti.ui.Camera()
cam.up(x=0, y=1, z=0)
cam.lookat(x=0.0, y=0.0, z=0.0)
cam.position(x=13.0, y=2.0, z=3.0)
cam.fov(20)

vertex_field = Vector.field(shape=(2,))
vertex_field[0] = Vector((1.0, 0.0, 1.0))
vertex_field[1] = Vector((0.0, 0.5, 0.0))
k = 0
old_value = 5

while window.running:
    render()
    camera = Camera(
        cam.curr_position,
        cam.curr_lookat,
        Vector(0, 1, 0),
        20,
        aspect_ratio,
        aperture,
        focus_distance,
    )
    # with gui.sub_window(name="Parameters", x=0.0, y=0.0, width=0.3, height=0.3) as g:
    # level = g.slider_int(
    #     text="BVH Level", old_value=old_value, maximum=len(world.bvh_nodes) * 24, minimum=1
    # )
    # old_value = level

    scene.set_camera(cam)
    # n = world.vertex_bvh.to_numpy()[:level]
    # world.vertex_bvh.fill(Vector(0.0))
    # world.vertex_bvh.from_numpy(n)
    scene.particles(centers=world.vertex_bvh, color=(0.68, 0.26, 0.19), radius=0.03)
    scene.lines(vertices=world.vertex_bvh, width=0.1)
    scene.ambient_light((0.8, 0.8, 0.8))
    canvas.scene(scene)
    canvas.set_image((pixels.to_numpy() / k) ** 0.5)
    window.show()
    k += 1
