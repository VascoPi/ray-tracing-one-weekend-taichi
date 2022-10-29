import taichi as ti
from vector import *

ti.init(arch=ti.gpu)

width, height = 256, 256
pixels = ti.Vector.field(n=3, dtype=ti.f32, shape=(width, height))

@ti.kernel
def paint():
    for i, j in pixels:
        pixels[i, j] = Color(i / (width + 1), j / (height + 1), 0.25)

gui = ti.GUI(name='Render', res=(width, height), show_gui=True)
paint()
gui.set_image(pixels)
gui.show("output.png")

