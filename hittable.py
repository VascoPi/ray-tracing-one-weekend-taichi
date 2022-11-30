import taichi as ti
from vector import Point, Color
from material import MaterialData, Dielectric, Metal, Lambert
import random
from bvh import build_bvh, BVHNode, hit_aabb, set_next_id_links, surrounding_box
from vector import Vector


def random_scene(is_bvh=False):
    world = HittableList(is_bvh=is_bvh)

    material1 = Dielectric(1.5)
    material2 = Lambert(Color(0.4, 0.2, 0.1))
    material3 = Metal(Color(0.7, 0.6, 0.5), 0.0)

    world.add(Sphere(center=Point(0.0, 1.0, 0.0), radius=1.0, material=material1))
    world.add(Sphere(center=Point(-4.0, 1.0, 0.0), radius=1.0, material=material2))
    world.add(Sphere(center=Point(4.0, 1.0, 0.0), radius=1.0, material=material3))

    for a in range(-3, 3):
        for b in range(-3, 3):
            material_type = random.random()
            center = Point(a + 0.9 * random.random(), 0.2, b + 0.9 * random.random())
            if center - Point(4.0, 0.2, 0.0).norm() > 0.9:
                if material_type < 0.8:
                    material = Lambert(Color(random.random(), random.random(), random.random()))
                elif material_type < 0.95:
                    material = Metal(
                        Color(random.random(), random.random(), random.random()) * 0.5 + 0.5,
                        random.random() * 0.5,
                    )
                else:
                    material = Dielectric(1.5)

            world.add(Sphere(center=center, radius=0.2, material=material))

    material_ground = Lambert(Color(0.5, 0.5, 0.5))
    world.add(Sphere(center=Point(0.0, -1000.0, 0.0), radius=1000.0, material=material_ground))

    return world


@ti.data_oriented
class HittableList:
    def __init__(self, is_bvh=False):
        self.objects = []
        self.is_bvh = is_bvh

    def add(self, object):
        # set the id and add to list

        object.id = len(self.objects)
        self.objects.append(object)

    def commit(self):
        """Save the sphere data and material info so we can loop over these."""
        self.n = len(self.objects)
        self.sphere_infos = SphereInfo.field(shape=(self.n,))
        self.mat_infos = MaterialData.field(shape=(self.n,))
        self.vertex = Vector.field(shape=self.n * 24)
        bvh_nodes = build_bvh(self.objects)
        self.bvh = self.build(bvh_nodes)
        self.vertex_bvh = Vector.field(shape=len(bvh_nodes) * 24)

        for i, sphere in enumerate(self.objects):
            sphere_info, mat_info = sphere.get_info()
            self.sphere_infos[i] = sphere_info
            self.mat_infos[i] = mat_info

        # self.vertex[0] = Vector(1, 1, 1)
        # self.vertex[1] = Vector(0, 1, 1)
        #
        #
        # self.vertex[2] = Vector(1, 1, 0)
        # self.vertex[3] = Vector(0, 1, 0)
        #
        #
        # self.vertex[4] = Vector(1, 0, 1)
        # self.vertex[5] = Vector(0, 0, 1)
        #
        #
        # self.vertex[6] = Vector(1, 0, 0)
        # self.vertex[7] = Vector(0, 0, 0)
        #
        #
        # self.vertex[8] = Vector(1, 1, 1)
        # self.vertex[9] = Vector(1, 0, 1)
        #
        #
        # self.vertex[10] = Vector(1, 1, 0)
        # self.vertex[11] = Vector(1, 0, 0)
        #
        #
        # self.vertex[12] = Vector(0, 1, 1)
        # self.vertex[13] = Vector(0, 0, 1)
        #
        #
        # self.vertex[14] = Vector(0, 1, 0)
        # self.vertex[15] = Vector(0, 0, 0)
        #
        #
        # self.vertex[16] = Vector(1, 1, 1)
        # self.vertex[17] = Vector(1, 1, 0)
        #
        #
        # self.vertex[18] = Vector(1, 0, 1)
        # self.vertex[19] = Vector(1, 0, 0)
        #
        #
        # self.vertex[20] = Vector(0, 1, 1)
        # self.vertex[21] = Vector(0, 1, 0)
        #
        #
        # self.vertex[22] = Vector(0, 0, 1)
        # self.vertex[23] = Vector(0, 0, 0)

        # for k, sphere in enumerate(self.objects):
        #     idx = k * 24
        #     bounding_box_verts(idx, self.vertex, sphere)

        for k, aabb in enumerate(bvh_nodes):
            idx = k * 24
            if aabb.obj_id == -1 and aabb.level <= 6:
                bounding_box_verts(idx, self.vertex_bvh, aabb)

            # bounding_box_verts(idx, self.vertex_bvh, aabb)

    def build(self, bvh_nodes):
        set_next_id_links(bvh_nodes)
        bvh_field = BVHNode.field(shape=(len(bvh_nodes),))
        for i, v in enumerate(bvh_nodes):
            bvh_field[i] = v

        return bvh_field

    # @ti.func
    # def hit(self, r, t_min, t_max):
    #     hit_anything = False
    #     rec = HitRecord()
    #     mat_info = MaterialData()
    #     # hit_anything, rec, mat_info = self.hit_nobvh(r, t_min, t_max)
    #     if self.is_bvh:
    #         hit_anything, rec, mat_info = self.hit_bvh(r, t_min, t_max)
    #     else:
    #         hit_anything, rec, mat_info = self.hit_nobvh(r, t_min, t_max)
    #
    #     return hit_anything, rec, mat_info

    @ti.func
    def hit(self, r, t_min, t_max):
        hit_anything = False
        closest_so_far = t_max
        rec = HitRecord()
        mat_info = MaterialData()
        curr_id = 0
        c = 0

        while curr_id != -1:
            c += 1
            # print(c)
            bvh_node = self.bvh[curr_id]
            if bvh_node.obj_id != -1:
                hit, temp_rec = Sphere.hit(
                    self.sphere_infos[bvh_node.obj_id], r, t_min, closest_so_far
                )
                if hit:
                    hit_anything = True
                    closest_so_far = temp_rec.t
                    rec = temp_rec
                    # we return the material info not the material because
                    # taichi doesn't yet deal with assigning object pointers
                    mat_info = self.mat_infos[bvh_node.obj_id]

                # print("Sphere Finish", "curr_id:", curr_id, "next_id:", bvh_node.next_id, "hit:", hit)
                curr_id = bvh_node.next_id

            else:
                # print("AABB Start", "curr_id:", curr_id, "next_id:", bvh_node.next_id)
                hit = hit_aabb(bvh_node, r, t_min, closest_so_far)
                if hit:
                    curr_id = bvh_node.left_id
                else:
                    curr_id = bvh_node.next_id
                # print("AABB Finish", "curr_id:", curr_id, "next_id:", bvh_node.next_id)
        # print("AABB Out", "curr_id:", curr_id)
        return hit_anything, rec, mat_info

    @ti.func
    def hit_(self, r, t_min, t_max):
        hit_anything = False
        closest_so_far = t_max
        rec = HitRecord()
        mat_info = MaterialData()
        for i in range(self.n):
            hit, temp_rec = Sphere.hit(self.sphere_infos[i], r, t_min, closest_so_far)
            if hit:
                hit_anything = True
                closest_so_far = temp_rec.t
                rec = temp_rec
                # we return the material info not the material because
                # taichi doesn't yet deal with assigning object pointers
                mat_info = self.mat_infos[i]

        return hit_anything, rec, mat_info


SphereInfo = ti.types.struct(center=Point, radius=ti.f32)


@ti.data_oriented
class Sphere:
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material
        self.box_min = Vector(center - Vector(radius))
        self.box_max = Vector(center + Vector(radius))

    def get_info(self):
        return (
            SphereInfo(
                center=self.center, radius=self.radius, box_min=self.box_min, box_max=self.box_max
            ),
            self.material.mat_info,
        )

    def get_bounds(self):
        return Vector(self.center - Vector(self.radius)), Vector(self.center + Vector(self.radius))

    @staticmethod
    @ti.func
    def hit(self, ray, t_min, t_max):
        result = False
        hit_record = HitRecord()
        oc = ray.origin - self.center
        a = ray.direction.norm_sqr()
        half_b = oc.dot(ray.direction)
        c = oc.norm_sqr() - self.radius**2
        discriminant = half_b**2 - a * c
        if discriminant >= 0.0:
            sqrtd = ti.sqrt(discriminant)
            root = (-half_b - sqrtd) / a
            if root < t_min or t_max < root:
                root = (-half_b + sqrtd) / a

            result = not (root < t_min or t_max < root)

            if result:
                hit_record.t = root
                hit_record.point = ray.at(hit_record.t)
                outward_normal = (hit_record.point - self.center) / self.radius
                set_face_normal(ray, outward_normal, hit_record)

        return result, hit_record


@ti.dataclass
class HitRecord:
    point: Point
    normal: Color
    t: ti.f32
    mat_info: MaterialData
    front_face: ti.i32


@ti.func
def set_face_normal(ray, outward_normal, hit_record: ti.template()):
    hit_record.front_face = ray.direction.dot(outward_normal) < 0
    hit_record.normal = outward_normal if hit_record.front_face else -outward_normal


def bounding_box_verts(idx, field, obj):
    if hasattr(obj, "get_bounds"):
        box_min, box_max = obj.get_bounds()
    else:
        box_min, box_max = obj.box_min, obj.box_max

    field[idx + 0] = Vector(box_max[0], box_max[1], box_max[2])
    field[idx + 1] = Vector(box_min[0], box_max[1], box_max[2])

    field[idx + 2] = Vector(box_max[0], box_max[1], box_min[2])
    field[idx + 3] = Vector(box_min[0], box_max[1], box_min[2])

    field[idx + 4] = Vector(box_max[0], box_min[1], box_max[2])
    field[idx + 5] = Vector(box_min[0], box_min[1], box_max[2])

    field[idx + 6] = Vector(box_max[0], box_min[1], box_min[2])
    field[idx + 7] = Vector(box_min[0], box_min[1], box_min[2])

    field[idx + 8] = Vector(box_max[0], box_max[1], box_max[2])
    field[idx + 9] = Vector(box_max[0], box_min[1], box_max[2])

    field[idx + 10] = Vector(box_max[0], box_max[1], box_min[2])
    field[idx + 11] = Vector(box_max[0], box_min[1], box_min[2])

    field[idx + 12] = Vector(box_min[0], box_max[1], box_max[2])
    field[idx + 13] = Vector(box_min[0], box_min[1], box_max[2])

    field[idx + 14] = Vector(box_min[0], box_max[1], box_min[2])
    field[idx + 15] = Vector(box_min[0], box_min[1], box_min[2])

    field[idx + 16] = Vector(box_max[0], box_max[1], box_max[2])
    field[idx + 17] = Vector(box_max[0], box_max[1], box_min[2])

    field[idx + 18] = Vector(box_max[0], box_min[1], box_max[2])
    field[idx + 19] = Vector(box_max[0], box_min[1], box_min[2])

    field[idx + 20] = Vector(box_min[0], box_max[1], box_max[2])
    field[idx + 21] = Vector(box_min[0], box_max[1], box_min[2])

    field[idx + 22] = Vector(box_min[0], box_min[1], box_max[2])
    field[idx + 23] = Vector(box_min[0], box_min[1], box_min[2])
