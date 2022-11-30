import taichi as ti
import copy
from vector import Vector, get_bounding_box, Color


@ti.func
def hit_aabb(bvh_node, r, t_min, t_max):
    intersect = True
    min_aabb, max_aabb = bvh_node.box_min, bvh_node.box_max
    ray_direction, ray_origin = r.direction, r.origin

    for i in ti.static(range(3)):
        if ray_direction[i] == 0:
            if ray_origin[i] < min_aabb[i] or ray_origin[i] > max_aabb[i]:
                intersect = False
        else:
            i1 = (min_aabb[i] - ray_origin[i]) / ray_direction[i]
            i2 = (max_aabb[i] - ray_origin[i]) / ray_direction[i]

            new_t_max = ti.max(i1, i2)
            new_t_min = ti.min(i1, i2)

            t_max = ti.min(new_t_max, t_max)
            t_min = ti.max(new_t_min, t_min)

    if t_min > t_max:
        intersect = False
    return intersect


BVHNode = ti.types.struct(box_min=Vector, box_max=Vector,
                          obj_type=ti.i32, obj_id=ti.i32,
                          left_id=ti.i32, right_id=ti.i32,
                          parent_id=ti.i32, next_id=ti.i32, color=Color, level=ti.i32)


def set_next_id_links(bvh_node_list):
    ''' given a list of nodes set the 'next_id' link in the nodes '''
    def inner_loop(node_id):
        node = bvh_node_list[node_id]
        if node.parent_id == -1:
            return -1

        parent = bvh_node_list[node.parent_id]
        if parent.right_id != -1 and parent.right_id != node_id:
            return parent.right_id
        else:
            return inner_loop(node.parent_id)

    for i, node in enumerate(bvh_node_list):
        node.next_id = inner_loop(i)


def sort_obj_list(obj_list):
    ''' Sort the list of objects along the longest directional span '''
    def get_x(e):
        obj = e
        return obj.center.x

    def get_y(e):
        obj = e
        return obj.center.y

    def get_z(e):
        obj = e
        return obj.center.z

    centers = [obj.center for obj in obj_list]
    min_center = [
        min([center[0] for center in centers]),
        min([center[1] for center in centers]),
        min([center[2] for center in centers])
    ]
    max_center = [
        max([center[0] for center in centers]),
        max([center[1] for center in centers]),
        max([center[2] for center in centers])
    ]
    span_x, span_y, span_z = (max_center[0] - min_center[0],
                              max_center[1] - min_center[1],
                              max_center[2] - min_center[2])
    if span_x >= span_y and span_x >= span_z:
        obj_list.sort(key=get_x)
    elif span_y >= span_z:
        obj_list.sort(key=get_y)
    else:
        obj_list.sort(key=get_z)
    return obj_list


def build_bvh(object_list, parent_id=-1, curr_id=0, level=1, color=Color(1.0, 0.0, 0.0)):
    obj_list = copy.copy(object_list)
    span = len(obj_list)
    bvh_nodes = []
    box_min, box_max = get_bounding_box(obj_list[0])
    if span == 1:
        bvh_nodes.append(BVHNode(box_min=box_min, box_max=box_max,
                            left_id=-1, right_id=-1,
                            parent_id=parent_id,
                            next_id=-1,
                            obj_id=obj_list[0].id,
                            color=color, level=level))

    else:
        sorted_list = sort_obj_list(obj_list)
        mid = int(span/2)
        left_nodelist = build_bvh(sorted_list[:mid], parent_id=curr_id, curr_id=curr_id + 1, level=level+1, color=Color(1.0, 0.0, 0.0))
        right_nodelist = build_bvh(sorted_list[mid:], parent_id=curr_id, curr_id=curr_id + len(left_nodelist) + 1, level=level+1, color=Color(0.0, 0.0, 1.0))

        box_min, box_max, = surrounding_box(
            (left_nodelist[0].box_min, left_nodelist[0].box_max),
            (right_nodelist[0].box_min, right_nodelist[0].box_max))

        bvh_nodes.append(BVHNode(box_min=box_min, box_max=box_max,
                                left_id=curr_id + 1, right_id=curr_id + len(left_nodelist) + 1,
                                parent_id=parent_id,
                                next_id=-1,
                                obj_id=-1,
                                color=color,
                                level=level))
        bvh_nodes = bvh_nodes + left_nodelist + right_nodelist

    return bvh_nodes


def surrounding_box(box1, box2):
    ''' Calculates the surround bbox of two bboxes '''
    box1_min, box1_max = box1
    box2_min, box2_max = box2

    small = ti.min(box1_min, box2_min)
    big = ti.max(box1_max, box2_max)
    return small, big
