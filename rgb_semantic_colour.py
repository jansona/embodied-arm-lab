"""Shared conservative RGB semantic-colour gate for the narrow cuboid profile.

Hue is an additional intersection with existing dominance/chroma/brightness
requirements, never a relaxed threshold. It prevents yellow/orange robot parts
from being called red merely because their R channel exceeds G. This is an
explicit known-colour appearance assumption, not a general object detector.
No supplied masks, depth, geometry, simulator state, or learned confidence enter.
"""
import numpy as np


def semantic_colour_mask(rgb, object_id):
    rgb = np.asarray(rgb, dtype=float)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or not np.isfinite(rgb).all():
        raise ValueError('Expected finite RGB image')
    if object_id not in ('red', 'blue', 'goal'):
        raise ValueError('Unknown semantic colour')
    r, g, b = np.moveaxis(rgb, 2, 0)
    index = {'red':0, 'blue':2, 'goal':1}[object_id]
    channels = (r, g, b)
    dominant = channels[index]
    other_a, other_b = [channels[i] for i in range(3) if i != index]
    old = ((dominant >= 55) & (dominant-(other_a+other_b)/2 >= 32)
           & (dominant > 1.25*other_a) & (dominant > 1.25*other_b))
    maximum = np.max(rgb, axis=2)
    minimum = np.min(rgb, axis=2)
    delta = maximum - minimum
    divisor = np.where(delta > 0, delta, 1.)
    hue = np.zeros(delta.shape)
    red = (maximum == r) & (delta > 0)
    green = (maximum == g) & ~red & (delta > 0)
    blue = ~(red | green) & (delta > 0)
    hue[red] = ((g-b)/divisor)[red] % 6
    hue[green] = ((b-r)/divisor)[green] + 2
    hue[blue] = ((r-g)/divisor)[blue] + 4
    hue *= 60
    semantic = {'red': (hue <= 20) | (hue >= 340),
                'goal': (hue >= 80) & (hue <= 160),
                'blue': (hue >= 200) & (hue <= 260)}[object_id]
    return old & semantic
