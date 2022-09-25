import sys

# setting path
sys.path.append('./')

from .pixelcoords import PixelCoords

# Stores pixel vector between two PixelCoords
class PixelVector:
    x_vector = None
    y_vector = None
    # Note that length here is measured in microns (μm)
    x_length = None
    y_length = None

    def __init__(self, from_coords, to_coords, pixel_length=1.55):
        self.x_vector = to_coords.x - from_coords.x 
        self.y_vector = to_coords.y - from_coords.x
        self.x_length = self.x_vector*pixel_length
        self.y_length = self.y_vector*pixel_length