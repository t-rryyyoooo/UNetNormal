import argparse
import SimpleITK as sitk
import numpy as np
import sys
sys.path.append("..")
from utils.utils import sitkReadImageElseNone, getSizeFromStringElseNone
from imageSlicer import ImageSlicer

def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_image_path", help="case_00/imaging_1.nii.gz")
    parser.add_argument("target_image_path", help="case_00/imaging_2.nii.gz")
    parser.add_argument("save_dir", help="patch")
    parser.add_argument("patient_id", help="00")
    parser.add_argument("--mask_image_path", help="case_00/mask.nii.gz")
    parser.add_argument("--input_patch_width", default=1, type=int)
    parser.add_argument("--target_patch_width", default=1, type=int)
    parser.add_argument("--plane_size", help="default: None (means orignal slice)")
    parser.add_argument("--overlap", help="", type=int, default=1)
    parser.add_argument("--axis", type=int, help="Image arrayis sliced perpendicular to it. [ex] 0", default=0)
    parser.add_argument("--input_name", help="Saved input patch name.", default="input")
    parser.add_argument("--target_name", help="Saved target patch name.", default="target")
    parser.add_argument("--with_nonmask", action="store_true")

    args = parser.parse_args()

    return args

def main(args):
    input_image  = sitk.ReadImage(args.input_image_path)
    target_image = sitk.ReadImage(args.target_image_path)
    mask_image   = sitkReadImageElseNone(args.mask_image_path)

    plane_size = getSizeFromStringElseNone(args.plane_size, digit=2)

    image_slicer = ImageSlicer(
                    image              = input_image,
                    target             = target_image,
                    image_patch_width  = args.input_patch_width,
                    target_patch_width = args.target_patch_width,
                    plane_size         = plane_size,
                    overlap            = args.overlap,
                    axis               = args.axis,
                    mask               = mask_image
                    )

    image_slicer.savePatchArray(
                    args.save_dir, 
                    args.patient_id, 
                    input_name   = args.input_name, 
                    target_name  = args.target_name,
                    with_nonmask = args.with_nonmask
                    )

if __name__ == "__main__":
    args = parseArgs()
    main(args)
