import argparse
from pathlib import Path
import SimpleITK as sitk
import numpy as np
from extractor import Extractor
from utils.utils import getSizeFromString

args = None

def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("image_path", help="$HOME/Desktop/data/kits19/case_00000/imaging.nii.gz")
    parser.add_argument("label_path", help="$HOME/Desktop/data/kits19/case_00000/segmentation.nii.gz")
    parser.add_argument("save_path", help="$HOME/Desktop/data/slice/hist_0.0/case_00000", default=None)
    parser.add_argument("patient_id", help="For makign save_path")
    parser.add_argument("--mask_path", help="$HOME/Desktop/data/kits19/case_00000/label.mha")
    parser.add_argument("--image_patch_size", help="48-48-16", default="48-48-16")
    parser.add_argument("--label_patch_size", help="48-48-16", default="48-48-16")
    parser.add_argument("--overlap", help="1", type=int, default=1)
    parser.add_argument("--num_class", help="1", type=int, default=14)
    parser.add_argument("--class_axis", help="1", type=int, default=0)
    parser.add_argument("--with_nonmask", action="store_true")

    args = parser.parse_args()
    return args

def main(args):
    """ Read image and label. """
    label = sitk.ReadImage(args.label_path)
    image = sitk.ReadImage(args.image_path)
    if args.mask_path is not None:
        mask = sitk.ReadImage(args.mask_path)
    else:
        mask = None

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

    #extractor.save(args.save_path, args.patient_id, with_nonmask=args.with_nonmask)

    # For testing iace.outputRestoredImage.
    from tqdm import tqdm
    with tqdm(total=extractor.__len__(), ncols=60, desc="Segmenting and restoring...") as pbar:
        for ipa, lpa, mpa, _, index in extractor.generateData():
            mpa = sitk.GetArrayFromImage(mpa)
            lpa = sitk.GetArrayFromImage(lpa)
            if (mpa > 0).any():
                lpa_onehot = np.eye(args.num_class)[lpa].transpose(3, 0, 1, 2) 
                extractor.insertToPredictedArray(index, lpa_onehot)

            pbar.update(1)

    predicted = extractor.outputRestoredImage()
    sitk.WriteImage(predicted, "/Users/tanimotoryou/Desktop/label.mha")
    pa = sitk.GetArrayFromImage(predicted)
    la = sitk.GetArrayFromImage(label)
    from utils.indicator.DICE import DICE
    dice = DICE(la, pa)
    print(dice)

if __name__ == '__main__':
    args = ParseArgs()
    main(args)
