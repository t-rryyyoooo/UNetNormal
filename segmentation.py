import SimpleITK as sitk
import sys
sys.path.append("..")
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import pickle5
from utils.machineLearning.predict import Predictor as Segmenter
from utils.utils import getSizeFromStringElseNone, isMasked, sitkReadImageElseNone


def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("image_path", help="$HOME/Desktop/data/kits19/case_00000/imaging.nii.gz")
    parser.add_argument("modelweightfile", help="Trained model weights file (*.hdf5).")
    parser.add_argument("save_path", help="Segmented label file.(.mha)")
    parser.add_argument("--model_type", help="[2dunet512/3dunet]")
    parser.add_argument("--mask_path", help="$HOME/Desktop/data/kits19/case_00000/mask.mha")
    parser.add_argument("--image_patch_width", default=1, type=int)
    parser.add_argument("--label_patch_width", default=1, type=int)
    parser.add_argument("--plane_size")
    parser.add_argument("--image_patch_size", help="48-48-16", default="44-44-28")
    parser.add_argument("--label_patch_size", help="44-44-28", default="44-44-28")
    parser.add_argument("--overlap", help="1", default=1, type=int)
    parser.add_argument("--num_class", help="14", default=14, type=int)
    parser.add_argument("--axis", help="0", default=0, type=int)
    parser.add_argument("-g", "--gpuid", help="0 1", nargs="*", default=0, type=int)

    args = parser.parse_args()
    return args

def main(args):
    """ Read images. """
    image = sitk.ReadImage(args.image_path)
    mask  = sitkReadImageElseNone(args.mask_path)

    """ Dummy image for prediction"""
    label = sitk.Image(image.GetSize(), sitk.sitkInt8)
    label.SetOrigin(image.GetOrigin())
    label.SetDirection(image.GetDirection())
    label.SetSpacing(image.GetSpacing())


    if args.model_type not in ["2dunet512", "3dunet"]:
        raise NotImplementedError("{} is not supported.".format(args.model_type))

    elif args.model_type == "3dunet":
        from extractor import Extractor
        from model.UNet_no_pad_with_nonmask.transform import UNetTransform as transform

        """ Get the patch size from string."""
        image_patch_size = getSizeFromStringElseNone(args.image_patch_size)
        label_patch_size = getSizeFromStringElseNone(args.label_patch_size)

        patch_generator = Extractor(
                            image            = image, 
                            label            = label, 
                            mask             = mask,
                            image_patch_size = image_patch_size, 
                            label_patch_size = label_patch_size, 
                            overlap          = args.overlap, 
                            num_class        = args.num_class,
                            class_axis       = args.axis
                        )

    else:
        from imageSlicer import ImageSlicer
        from model.UNet_2d_with_nonmask.transform import UNetTransform as transform

        plane_size = getSizeFromStringElseNone(args.plane_size, digit=2)
        patch_generator = ImageSlicer(
                            image              = image,
                            target             = label,
                            image_patch_width  = args.image_patch_width,
                            target_patch_width = args.label_patch_width,
                            plane_size         = plane_size,
                            overlap            = args.overlap,
                            axis               = args.axis,
                            mask               = mask
                            )

                            

    """ Load model. """
    with open(args.modelweightfile, 'rb') as f:
        model = pickle5.load(f)
        # Use a single gpu because It does't work when we use multi gpu.
        model = torch.nn.DataParallel(model, device_ids=[args.gpuid[0]])
    model.eval()

    """ Segmentation module. """
    use_cuda = torch.cuda.is_available() and True
    device = torch.device("cuda" if use_cuda else "cpu")
    segmenter = Segmenter(
                    model,
                    device = device
                    )

    transformer = transform()
    with tqdm(total=patch_generator.__len__(), ncols=60, desc="Segmenting and restoring...") as pbar:
        for image_patch, _, mask_patch, index in patch_generator.generatePatchArray():
            image_patch_array, mask_patch_array = transformer("test", image_patch, mask_patch)
            if isMasked(mask_patch_array):
                segmented_array = segmenter(image_patch_array)
                segmented_array = np.argmax(segmented_array, axis=0)

                patch_generator.insertToPredictedArray(index, segmented_array)

            pbar.update(1)

    segmented = patch_generator.outputRestoredImage()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print("Saving image to {}".format(str(save_path)))
    sitk.WriteImage(segmented, str(save_path), True)

if __name__ == '__main__':
    args = ParseArgs()
    main(args)
    
