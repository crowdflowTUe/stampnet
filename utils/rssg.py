import numpy as np
from utils.seed_util import retrieve_random_state as rrs
import skimage.draw as skdraw


def my_square(center, imsize, shape_size, radius=None):
    radius = shape_size if radius is None else radius
    return skdraw.polygon(
        np.array([center[0] - radius, center[0] + radius,
                  center[0] + radius, center[0] - radius]),
        np.array([center[1] - radius, center[1] - radius,
                  center[1] + radius, center[1] + radius]),
        shape=imsize
    )


def my_plus(center, imsize, shape_size, radius=None, width=.3):

    radius = shape_size if radius is None else radius
    w_size = int(shape_size * width)

    p1 = skdraw.polygon(
        np.array([center[0] - radius, center[0] + radius,
                  center[0] + radius, center[0] - radius]),
        np.array([center[1] - w_size, center[1] - w_size,
                  center[1] + w_size, center[1] + w_size]),
        shape=imsize
    )

    p2 = skdraw.polygon(
        np.array([center[0] - w_size, center[0] + w_size,
                  center[0] + w_size,
                  center[0] - w_size]),
        np.array([center[1] - radius, center[1] - radius,
                  center[1] + radius, center[1] + radius]),
        shape=imsize
    )

    return (
        np.concatenate((p1[0], p2[0])), np.concatenate((p1[1], p2[1]))
    )


def my_equal(center, imsize, shape_size, radius=None, width=.3, offset=.5):

    radius = shape_size if radius is None else radius
    w_size = int(shape_size * width)
    tb_offset = int(shape_size * offset)

    p1 = skdraw.polygon(
        np.array([center[0] - w_size + tb_offset,
                  center[0] + w_size + tb_offset,
                  center[0] + w_size + tb_offset,
                  center[0] - w_size + tb_offset]),
        np.array([center[1] - radius, center[1] - radius,
                  center[1] + radius, center[1] + radius]),
        shape=imsize
    )

    p2 = skdraw.polygon(
        np.array([center[0] - w_size - tb_offset,
                  center[0] + w_size - tb_offset,
                  center[0] + w_size - tb_offset,
                  center[0] - w_size - tb_offset]),
        np.array([center[1] - radius, center[1] - radius,
                  center[1] + radius, center[1] + radius]),
        shape=imsize
    )

    return (
        np.concatenate((p1[0], p2[0])), np.concatenate((p1[1], p2[1]))
    )


def my_doublevertical(center,
                      imsize,
                      shape_size,
                      radius=None,
                      width=.3,
                      offset=.5):

    radius = shape_size if radius is None else radius
    w_size = int(shape_size * width)
    tb_offset = int(shape_size * offset)

    p1 = skdraw.polygon(
        np.array([center[1] - radius, center[1] - radius,
                  center[1] + radius, center[1] + radius]),
        np.array([center[0] - w_size + tb_offset,
                  center[0] + w_size + tb_offset,
                  center[0] + w_size + tb_offset,
                  center[0] - w_size + tb_offset]),
        shape=imsize
    )

    p2 = skdraw.polygon(
        np.array([center[1] - radius, center[1] - radius,
                  center[1] + radius, center[1] + radius]),
        np.array([center[0] - w_size - tb_offset,
                  center[0] + w_size - tb_offset,
                  center[0] + w_size - tb_offset,
                  center[0] - w_size - tb_offset]),
        shape=imsize
    )

    return (
        np.concatenate((p1[0], p2[0])), np.concatenate((p1[1], p2[1]))
    )


def my_empty_figure(center, imsize, size, radius=None):
    return [[], []]


def my_uptri(center, imsize, shape_size, radius=None, width=.45):
    radius = shape_size if radius is None else radius
    w_size = int(shape_size * width)

    p = skdraw.polygon(
        np.array([center[0] - w_size,
                  center[0] + 2*w_size,
                  center[0] + 2*w_size]),
        np.array([center[1] - w_size,
                  center[1] - w_size,
                  center[1] + 2*w_size])
    )

    return p


def my_divide(center, imsize, shape_size, radius=None, width=.3):
    radius = shape_size if radius is None else radius
    w_size = int(shape_size * width)

    p = skdraw.polygon(
        np.array([center[0] + radius - w_size,
                  center[0] + radius,
                  center[0] - radius + w_size,
                  center[0] - radius]),
        np.array([center[1] - radius,
                  center[1] - radius + w_size,
                  center[1] + radius,
                  center[1] + radius - w_size])
    )

    return p


def my_regular_polygon(center,
                       imsize,
                       shape_size,
                       radius=None,
                       rot=0.,
                       sides=3,
                       offset_x=0.):
    radius = shape_size if radius is None else radius
    degs__ = np.arange(sides)*2*np.pi/float(sides) + rot

    offset_x *= radius

    cc_pos = np.cos(degs__)
    rr_pos = np.sin(degs__)

    return skdraw.polygon(
        center[0] + radius*rr_pos + offset_x,
        center[1] + radius*cc_pos,
        shape=imsize
    )


my_triangle = lambda *args, **kw: my_regular_polygon(*args, sides=3, **kw)


available_shapes = {
    "triangle": my_triangle,
    "plus": my_plus,
    "equal": my_equal,
    "doublevertical": my_doublevertical,
    "empty": my_empty_figure,
    "uptri": my_uptri,
    "divide": my_divide,
    "square": my_square
}

def generate_random_image(target_image,
                          shape_size,
                          flush_out=True,
                          shape_idx=None,
                          shapes_to_use=None,
                          const_value=1.,
                          output_type="image",
                          random_seed=None):
    """
    Generates a random shape and returns the shape on target_image.
    The shapes_to_use indicate which shapes to use.
    While output_type is used to retrieve the coordindates as well if necessary.
    The value of the shape will be const_value
    """

    r = rrs(random_seed)

    # Retrieve the shapes to choose from.
    if shapes_to_use is None:
        shapes_to_use = list(available_shapes.values())
    else:
        shapes_to_use = [available_shapes[x] for x in shapes_to_use]

    if flush_out:
        target_image[:] = 0.

    if shape_idx is None:
        random_shape_idx = r.randint(0, len(shapes_to_use))
    else:
        random_shape_idx = shape_idx

    shape_generator = shapes_to_use[random_shape_idx]

    center = (
        r.randint(0 + shape_size//2,
                  target_image.shape[0] - shape_size//2 - shape_size % 2 + 1),
        r.randint(0 + shape_size//2,
                  target_image.shape[1] - shape_size//2 - shape_size % 2 + 1)
    )

    rr, cc = shape_generator(center,
                             target_image.shape,
                             shape_size=(shape_size // 2))

    target_image[rr, cc] = const_value

    if output_type == "center":
        return target_image, random_shape_idx, center
    elif output_type == "upper-left":
        return target_image, random_shape_idx, (center[0] - shape_size // 2,
                                                center[1] - shape_size // 2)
    elif output_type == "image":
        return target_image
