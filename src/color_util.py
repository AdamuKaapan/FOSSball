import colorsys


def hsv_to_rgb(hsv):
    """
    Converts the given HSV tuple to RGB values
    :param hsv: The input HSV tuple ((h, s, v) ranging from 0.0 to 1.0)
    :return: The converted RGB tuple ((r, g, b) ranging from 0 to 255)
    """
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2]))
