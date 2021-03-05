import sys
sys.path.append("..")
import re
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from tqdm import tqdm
from itertools import product
from utils.imageProcessing.clipping import clipping
from utils.imageProcessing.cropping import cropping
from utils.imageProcessing.padding import padding
from utils.patchGenerator.scanPatchGenerator import ScanPatchGenerator
from utils.patchGenerator.utils import calculatePaddingSize
from utils.utils import getImageWithMeta, isMasked

class Extractor():
    """
    Class which Clips the input image and label to patch size.
    In 13 organs segmentation, unlike kidney cancer segmentation,
    SimpleITK axis : [sagittal, coronal, axial] = [x, y, z]
    numpy axis : [axial, coronal, saggital] = [z, y, x]
    In this class we use simpleITK to clip mainly. Pay attention to the axis.
    
    """
    def __init__(self, image, label, mask=None, image_patch_size=[48, 48, 16], label_patch_size=[48, 48, 16], overlap=1, num_class=14, class_axis=0):
        """
        image : original CT image
        label : original label image
        mask : mask image of the same size as the label
        image_patch_size : patch size for CT image.
        label_patch_size : patch size for label image.
        slide : When clipping, shit the clip position by slide
        phase : train -> For training model, seg -> For segmentation

        """
        
        self.org = image
        self.image = image
        self.label = label
        if mask is not None:
            self.mask = mask

        else:
            self.mask = sitk.Image(image.GetSize(), sitk.sitkUInt8)
            self.mask = self.mask.__add__(1)

        """ patch_size = [z, y, x] """
        self.image_patch_size = np.array(image_patch_size)
        self.label_patch_size = np.array(label_patch_size)

        self.overlap = overlap
        self.slide = self.label_patch_size // overlap

        self.makeGenerator()

        """ After implementing makeGenerator(), self.label is padded to clip correctly. """
        self.num_class = num_class
        self.class_axis = class_axis
        self.predicted_array = np.zeros([num_class] + list(self.label.GetSize())[::-1], dtype=np.float)
        self.counter_array = np.ones(list(self.label.GetSize())[::-1], dtype=np.float)

    def makeGenerator(self):
        """ Calculate each padding size for label and image to clip correctly. """
        self.lower_pad_size, self.upper_pad_size = calculatePaddingSize(
                                                    np.array(self.label.GetSize()), 
                                                    self.image_patch_size, self.label_patch_size, 
                                                    self.slide
                                                    )

        """ Pad image, label and mask. """
        self.image = padding(
                        self.image, 
                        self.lower_pad_size[0].tolist(), 
                        self.upper_pad_size[0].tolist(), 
                        )
        self.label = padding(
                        self.label, 
                        self.lower_pad_size[1].tolist(), 
                        self.upper_pad_size[1].tolist()
                        )
        self.mask = padding(
                        self.mask, 
                        self.lower_pad_size[1].tolist(), 
                        self.upper_pad_size[1].tolist()
                        )

        self.image_patch_generator = ScanPatchGenerator(
                                            self.image,
                                            self.image_patch_size,
                                            self.slide
                                            )
        
        self.label_patch_generator = ScanPatchGenerator(
                                            self.label,
                                            self.label_patch_size,
                                            self.slide
                                            )
        
        self.mask_patch_generator = ScanPatchGenerator(
                                            self.mask,
                                            self.label_patch_size,
                                            self.slide
                                            )

    def __len__(self):
        return self.image_patch_generator.__len__()

    def generateData(self):
        """ [1] means patch array because PatchGenerator returns index and patch_array. """
        for ipa, lpa, mpa in zip(self.image_patch_generator(), self.label_patch_generator(), self.mask_patch_generator()):
            input_index = ipa[0]
            output_index = lpa[0]

            yield ipa[1], lpa[1], mpa[1], input_index, output_index

    def save(self, save_path, patient_id, with_nonmask=False):
        if not isinstance(patient_id, str):
            patient_id = str(patient_id)

        save_path = Path(save_path)
        save_mask_path = save_path / "mask" / "case_{}".format(patient_id.zfill(2))
        save_mask_path.mkdir(parents=True, exist_ok=True)

        if with_nonmask:
            save_nonmask_path = save_path / "nonmask" / "case_{}".format(patient_id.zfill(2))
            save_nonmask_path.mkdir(parents=True, exist_ok=True)

            desc = "Saving masked and nonmasked images and labels..."

        else:
            desc = "Saving masked images and labels..."

        with tqdm(total=self.image_patch_generator.__len__(), ncols=100, desc=desc) as pbar:
            for i, (ipa, lpa, mpa, _, _) in enumerate(self.generateData()):
                if isMasked(mpa):
                    save_masked_image_path = save_mask_path / "image_{:04d}.mha".format(i)
                    save_masked_label_path = save_mask_path / "label_{:04d}.mha".format(i)
                    sitk.WriteImage(ipa, str(save_masked_image_path), True)
                    sitk.WriteImage(lpa, str(save_masked_label_path), True)

                else:
                    if with_nonmask:
                        save_nonmasked_image_path = save_nonmask_path / "image_{:04d}.mha".format(i)
                        save_nonmasked_label_path = save_nonmask_path / "label_{:04d}.mha".format(i)

                        sitk.WriteImage(ipa, str(save_nonmasked_image_path), True)
                        sitk.WriteImage(lpa, str(save_nonmasked_label_path), True)

                pbar.update(1)

    def insertToPredictedArray(self, index, array):
        """ Insert predicted array (before argmax array) which has probability per class. """
        assert array.ndim == self.predicted_array.ndim

        predicted_slices = []
        s = slice(0, self.num_class)
        predicted_slices.append(s)
        counter_slices = []
        label_patch_array_size = self.label_patch_size[::-1]
        index = index[::-1]
        for i in range(array.ndim - 1):
            s = slice(index[i], index[i] + label_patch_array_size[i])

            predicted_slices.append(s)
            counter_slices.append(s)

        predicted_slices = tuple(predicted_slices)
        counter_slices = tuple(counter_slices)

        """ Array's shape and counter's array shape are not same, which leads to shape error, so, address it. """
        s = np.delete(np.arange(array.ndim), self.class_axis)
        s = np.array(array.shape)[s]

        self.predicted_array[predicted_slices] += array
        self.counter_array[counter_slices] += np.ones(s)
 

    def outputRestoredImage(self):
        """ Usually, this method is used after all of predicted patch array is insert to self.predicted_array with insertToPredictedArray. """

        """ Address division by zero. """
        self.counter_array = np.where(self.counter_array == 0, 1, self.counter_array)

        self.predicted_array /= self.counter_array
        self.predicted_array = np.argmax(self.predicted_array, axis=self.class_axis)
        predicted = getImageWithMeta(self.predicted_array, self.label)
        predicted = cropping(
                        predicted,
                        self.lower_pad_size[1].tolist(),
                        self.upper_pad_size[1].tolist()
                        )

        return predicted






