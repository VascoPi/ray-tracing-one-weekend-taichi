import taichi as ti
from .vector import Point
from .camera import Camera
from .material import scatter
from .vector import Color, Vector
from .hittable import random_scene
import click


@click.command()
@click.version_option()
def main():
    ti.init(arch=ti.gpu)

    # Image
    image_width = 800
    aspect_ratio = 3.0 / 2.0
    image_height = int(image_width / aspect_ratio)
    pixels = ti.Vector.field(n=3, dtype=ti.f32, shape=(image_width, image_height))
    samples_per_pixel = 500

    # Camera
    look_from = Point(13.0, 2.0, 3.0)
    look_at = Point(0.0, 0.0, 0.0)
    focus_distance = (look_from - look_at).norm()
    focus_distance = 10.0
    aperture = 0.1
    camera = Camera(look_from, look_at, Vector(0, 1, 0), 20, aspect_ratio, aperture, focus_distance)

    # World
    world = random_scene()
    world.commit()
    gui = ti.GUI(name="Render", res=(image_width, image_height))

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
            u = (i + ti.random()) / (image_width - 1)
            v = (j + ti.random()) / (image_height - 1)
            ray = camera.get_ray(u, v)
            pixels[i, j] += ray_color(ray, world)

    # render()
    for k in range(samples_per_pixel):
        render()

    ti.tools.imwrite((pixels.to_numpy() / samples_per_pixel) ** 0.5, "output.png")


if __name__ == "__main__":
    main(prog_name="TaichiRender")  # pragma: no cover
