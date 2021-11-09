from hittable.bvh import build
import taichi as ti
from vector import *
from .hit_record import empty_hit_record
from .sphere import sphere
from .sphere import hit as hit_sphere
from .moving_sphere import moving_sphere
from .moving_sphere import hit as hit_moving_sphere
from material import empty_material
from .obj_types import *
from .bvh import build, hit_aabb


@ti.data_oriented
class HittableList:
    def __init__(self):
        self.objects = {SPHERE: [], MOVING_SPHERE: []}

    def add(self, object):
        # set the id and add to list
        obj_type = get_object_type(object)
        object.id = len(self.objects[obj_type])
        self.objects[obj_type].append(object)

    def commit(self):
        ''' Save the sphere data so we can loop over these.'''
        self.n_static = len(self.objects[SPHERE])
        self.static_spheres = sphere.field(shape=(self.n_static,))

        self.n_moving = len(self.objects[MOVING_SPHERE])
        self.moving_spheres = moving_sphere.field(shape=(self.n_moving,))

        def fill_array(from_array, to_array):
            for i, obj in enumerate(from_array):
                to_array[i] = obj

        fill_array(self.objects[SPHERE], self.static_spheres)
        fill_array(self.objects[MOVING_SPHERE], self.moving_spheres)

        self.bvh = build(self.objects)

    @ti.func
    def hit(self, r, t_min, t_max):
        hit_anything = False
        closest_so_far = t_max
        rec = empty_hit_record()
        mat_info = empty_material()

        curr = 0

        # walk the bvh tree
        while curr != -1:
            bvh_node = self.bvh[curr]

            if bvh_node.obj_id != -1:
                # this is a leaf node, check the sphere
                hit, temp_rec, temp_mat_info = self.hit_obj(bvh_node.obj_type, bvh_node.obj_id, r,
                                                            t_min, closest_so_far)
                if hit:
                    hit_anything = True
                    closest_so_far = temp_rec.t
                    rec = temp_rec
                    mat_info = temp_mat_info
                curr = bvh_node.next_id
            else:
                if hit_aabb(bvh_node, r, t_min, closest_so_far):
                    # visit left child next (left child will visit it's next = right)
                    curr = bvh_node.left_id
                else:
                    curr = bvh_node.next_id

        return hit_anything, rec, mat_info

    @ti.func
    def hit_obj(self, obj_type, obj_id, r, t_min, t_max):
        hit = False
        rec = empty_hit_record()
        mat_info = empty_material()

        if obj_type == MOVING_SPHERE:
            hit, rec = hit_moving_sphere(self.moving_spheres[obj_id], r, t_min, t_max)
            mat_info = self.moving_spheres[obj_id].material
        else:
            hit, rec = hit_sphere(self.static_spheres[obj_id], r, t_min, t_max)
            mat_info = self.static_spheres[obj_id].material

        return hit, rec, mat_info