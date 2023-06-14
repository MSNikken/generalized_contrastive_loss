import random as rnd
import kornia.augmentation as K
import torchvision.transforms as ttf

class Snipper:
    def create_displacement(self, iou):
        # generates random relative displacement of unit square w.r.t other unit square to achieve desired iou
        X: float = rnd.random() * (1 - iou)
        Y: float = (X + iou - 1) / (X - 1)
        # Treat X and Y as symmetrical:
        if rnd.getrandbits(1):
            return Y, X
        return X, Y

    def generate_subarea_coords(self, area_width, area_height, sub_area_width, sub_area_height):
        x = rnd.random() * (area_width - sub_area_width)
        y = rnd.random() * (area_height - sub_area_height)
        return x, y, x + sub_area_width, y + sub_area_height

    def create_snippets(self, image, iou):
        disp_X, disp_Y = self.create_displacement(iou)
        im_width, im_height = image.size
        snippet_width = 0.5 * im_width
        snippet_height = 0.5 * im_height
        combine_width = snippet_width * (1 + abs(disp_X))
        combine_height = snippet_height * (1 + abs(disp_Y))
        x_left, y_top, x_right, y_bot = self.generate_subarea_coords(im_width, im_height, combine_width, combine_height)

        # Allow all configurations
        snippet1_pos = rnd.choice(['top_left', 'top_right', 'bottom_left', 'bottom_right'])
        if snippet1_pos == 'top_left':
            snippet1_coords = x_left, y_top, x_left + snippet_width, y_top + snippet_height
            snippet2_coords = x_left + disp_X * snippet_width, y_top + disp_Y * snippet_height, \
                              x_left + (1 + disp_X) * snippet_width, y_top + (1 + disp_Y) * snippet_height
        elif snippet1_pos == 'top_right':
            snippet1_coords = x_right - snippet_width, y_top, x_right, y_top + snippet_height
            snippet2_coords = x_right - (1 + disp_X) * snippet_width, y_top + disp_Y * snippet_height, \
                              x_right - disp_X * snippet_width, y_top + (1 + disp_Y) * snippet_height
        elif snippet1_pos == 'bottom_left':
            snippet1_coords = x_left, y_bot - snippet_height, x_left + snippet_width, y_bot
            snippet2_coords = x_left + disp_X * snippet_width, y_bot - (1 + disp_Y) * snippet_height, \
                              x_left + (1 + disp_X) * snippet_width, y_bot - disp_Y * snippet_height
        elif snippet1_pos == 'bottom_right':
            snippet1_coords = x_right - snippet_width, y_bot - snippet_height, x_right, y_bot
            snippet2_coords = x_right - (1 + disp_X) * snippet_width, y_bot - (1 + disp_Y) * snippet_height, \
                              x_right - disp_X * snippet_width, y_bot - disp_Y * snippet_height

        snippet1 = image.crop(snippet1_coords)
        snippet2 = image.crop(snippet2_coords)
        return snippet1, snippet2


class Augmentor:
    def __init__(self):
        self.augmentations = [K.RandomPlanckianJitter(p=0.8, mode='blackbody', keepdim=True),
                              K.ColorJiggle(p=0.5),
                              K.RandomPlasmaBrightness(p=0.5),
                              K.RandomPlasmaContrast(p=0.5),
                              K.RandomGrayscale(p=0.5),
                              K.RandomBoxBlur(p=0.5),
                              K.RandomChannelShuffle(p=0.5),
                              K.RandomMotionBlur(p=0.3, kernel_size=3, angle=3, direction=0.1),
                              K.RandomSolarize(p=0.5)]
        self.normalize = ttf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def apply_augmentations(self, images):
        for aug in self.augmentations:
            images = aug(images)
        return self.normalize(images)
