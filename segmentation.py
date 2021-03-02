import SimpleITK as sitk
import sys
sys.path.append("..")
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import cloudpickle
from extractor import Extractor
from utils.machineLearning.segmentation import Segmenter
from utils.utils import getSizeFromString, isMasked


def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("image_path", help="$HOME/Desktop/data/kits19/case_00000/imaging.nii.gz")
    parser.add_argument("modelweightfile", help="Trained model weights file (*.hdf5).")
    parser.add_argument("save_path", help="Segmented label file.(.mha)")
    parser.add_argument("--mask_path", help="$HOME/Desktop/data/kits19/case_00000/mask.mha")
    parser.add_argument("--image_patch_size", help="48-48-16", default="44-44-28")
    parser.add_argument("--label_patch_size", help="44-44-28", default="44-44-28")
    parser.add_argument("--overlap", help="1", default=1, type=int)
    parser.add_argument("--num_class", help="14", default=14, type=int)
    parser.add_argument("--class_axis", help="0", default=0, type=int)
    parser.add_argument("-g", "--gpuid", help="0 1", nargs="*", default=0, type=int)

    args = parser.parse_args()
    return args

def main(args):
    """ Read images. """
    image = sitk.ReadImage(args.image_path)
    if args.mask_path is not None:
        mask = sitk.ReadImage(args.mask_path)
    else:
        mask = None

    """ Dummy image for prediction"""
    label = sitk.Image(image.GetSize(), sitk.sitkInt8)
    label.SetOrigin(image.GetOrigin())
    label.SetDirection(image.GetDirection())
    label.SetSpacing(image.GetSpacing())

    """ Get the patch size from string."""
    image_patch_size = getSizeFromString(args.image_patch_size)
    label_patch_size = getSizeFromString(args.label_patch_size)

    extractor = Extractor(
                    image = image, 
                    label = label, 
                    mask = mask,
                    image_patch_size = image_patch_size, 
                    label_patch_size = label_patch_size, 
                    overlap = args.overlap, 
                    num_class = args.num_class,
                    class_axis = args.class_axis
                    )

    """ Load model. """
    with open(args.modelweightfile, 'rb') as f:
        model = cloudpickle.load(f)
        model = torch.nn.DataParallel(model, device_ids=args.gpuid)

    model.eval()

    """ Segmentation module. """
    use_cuda = torch.cuda.is_available() and True
    device = torch.device("cuda" if use_cuda else "cpu")
    segmenter = Segmenter(
                    model,
                    num_input_array = 1,
                    ndim = 5,
                    device = device
                    )

    with tqdm(total=extractor.__len__(), ncols=60, desc="Segmenting and restoring...") as pbar:
        for image_patch, _, mask_patch, _, index in extractor.generateData():
            image_patch_array = sitk.GetArrayFromImage(image_patch)
            mask_patch_array  = sitk.GetArrayFromImage(mask_patch)

            if isMasked(mask_patch_array):
                input_array_list = [image_patch_array]
                segmented_array = segmenter.forward(input_array_list)

                extractor.insertToPredictedArray(index, segmented_array)

            pbar.update(1)

    segmented = extractor.outputRestoredImage()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print("Saving image to {}".format(str(save_path)))
    sitk.WriteImage(segmented, str(save_path), True)

if __name__ == '__main__':
    args = ParseArgs()
    main(args)
    
