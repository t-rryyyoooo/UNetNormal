import SimpleITK as sitk
import numpy as np
from pathlib import Path
import torch

def setMeta(to_image, ref_image, direction=None, origin=None, spacing=None):
    if direction is None:
        direction = ref_image.GetDirection()

    if origin is None:
        origin = ref_image.GetOrigin()

    if spacing is None:
        spacing = ref_image.GetSpacing()

    to_image.SetDirection(direction)
    to_image.SetOrigin(origin)
    to_image.SetSpacing(spacing)

    return to_image

def separateData(image_path_list, label_path, criteria, phase): 
    dataset = []
    for number in criteria[phase]:
        image_lists = []
        for image_path in image_path_list:
            image_path = Path(image_path) / ("case_" + number) 

            image_list = image_path.glob("image*")
            image_list = sorted(image_list)
            image_lists.append(image_list)

        lab_path = Path(label_path) / ("case_" + number)
        label_list = lab_path.glob("label*")
        label_list = sorted(label_list)

        length = len(label_list)
        for image_list in image_lists:
            assert length == len(image_list)

        img_lists = []
        for image in zip(*image_lists):
            image = [str(x) for x in image]
            img_lists.append(image)

        for imgs, lab in zip(img_lists, label_list):
            dataset.append((imgs, str(lab)))

    return dataset


def makeAffineParameters(image, translate, rotate, shear, scale):
    dimension = image.GetDimension()
    translation = np.random.uniform(-translate, translate, dimension)
    rotation = np.radians(np.random.uniform(-rotate, rotate))
    shear = np.random.uniform(-shear, shear, 2)
    scale = np.random.uniform(1 - scale, 1 + scale)
    center = (np.array(image.GetSize()) * np.array(image.GetSpacing()) / 2)[::-1]

    return [translation, rotation, scale, shear, center]

def makeAffineMatrix(translate, rotate, scale, shear, center):
    a = sitk.AffineTransform(3)

    a.SetCenter(center)
    a.Rotate(1, 0, rotate)
    a.Shear(1, 0, shear[0])
    a.Shear(0, 1, shear[1])
    a.Scale((scale, scale, 1))
    a.Translate(translate)

    return a

def transforming(image, bspline, affine, interpolator, minval):
    # B-spline transformation
    if bspline is not None:
        transformed_b = sitk.Resample(image, bspline, interpolator, minval)

    # Affine transformation
        transformed_a = sitk.Resample(transformed_b, affine, interpolator, minval)

    else:
        transformed_a = sitk.Resample(image, affine, interpolator, minval)

    return transformed_a

def getMinimumValue(image):
    minmax = sitk.MinimumMaximumImageFilter()
    minmax.Execute(image)
    return minmax.GetMinimum()

class DICE():
    def __init__(self, num_class, device):
        self.num_class = num_class
        self.device = device
        """
        Required : not onehot
        """

    def compute(self, true, pred):
        eps = 10**-9
        assert true.size() == pred.size()
        
        true.to(self.device)
        true.to(self.device)

        
        intersection = (true * pred).sum()
        union = (true * true).sum() + (pred * pred).sum()
        dice = (2. * intersection) / (union + eps)
        """
        intersection = (true == pred).sum()
        union = (true != 0).sum() + (pred != 0).sum()
        dice = 2. * (intersection + eps) / (union + eps)
        """
        
        return dice

    def computePerClass(self, true, pred):
        DICE = []
        for x in range(self.num_class):
            true_part = (true == x).int()
            pred_part = (pred == x).int()
            """
            true_part = true[..., x]
            pred_part = pred[..., x]
            """
            dice = self.compute(true_part, pred_part)
            DICE.append(dice)

        return DICE

def cropping3D(image, crop_z, crop_x, crop_y):
    """
    image : only 5D tensor
    
    """
    size_z, size_x, size_y = image.size()[2:]
    crop_z = slice(crop_z[0], size_z - crop_z[1])
    crop_x = slice(crop_x[0], size_x - crop_x[1])
    crop_y = slice(crop_y[0], size_y - crop_y[1])
    
    cropped_image = image[..., crop_z, crop_x, crop_y]
    
    return cropped_image




