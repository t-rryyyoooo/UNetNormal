import SimpleITK as sitk
import numpy as np
import argparse
from functions import createParentPath, getImageWithMeta, getSizeFromString
from pathlib import Path
from extractor import Extractor
from tqdm import tqdm
import torch
import cloudpickle
import re
import sys
sys.path.append("..")


def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("image_path", help="$HOME/Desktop/data/kits19/case_00000/imaging.nii.gz")
    parser.add_argument("modelweightfile", help="Trained model weights file (*.hdf5).")
    parser.add_argument("save_path", help="Segmented label file.(.mha)")
    parser.add_argument("--mask_path", help="$HOME/Desktop/data/kits19/case_00000/mask.mha")
    parser.add_argument("--image_patch_size", help="48-48-16", default="44-44-28")
    parser.add_argument("--label_patch_size", help="44-44-28", default="44-44-28")
    parser.add_argument("--overlap", help="1", default=1, type=int)
    parser.add_argument("-g", "--gpuid", help="0 1", nargs="*", default=0, type=int)

    args = parser.parse_args()
    return args

def main(args):
    use_cuda = torch.cuda.is_available() and True
    device = torch.device("cuda" if use_cuda else "cpu")
    """ Slice module. """

    image = sitk.ReadImage(args.image_path)
    if args.mask_path is not None:
        mask = sitk.ReadImage(args.mask_path)
    else:
        mask = None

    """ Dummy image """
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
            )

    extractor.execute()

    """ Load model. """
    with open(args.modelweightfile, 'rb') as f:
        model = cloudpickle.load(f)
        model = torch.nn.DataParallel(model, device_ids=args.gpuid)

    model.eval()

    """ Segmentation module. """
    segmented_array_list = []
    for image, _ in tqdm(extractor.loadData(), desc="Segmenting images...", ncols=60):
        #image_array = image_array.transpose(2, 0, 1)
        image_array = sitk.GetArrayFromImage(image)
        image_array = torch.from_numpy(image_array[np.newaxis, np.newaxis, ...]).to(device, dtype=torch.float)

        segmented_array = model(image_array)
        segmented_array = segmented_array.to("cpu").detach().numpy().astype(np.float)
        segmented_array = np.squeeze(segmented_array)
        segmented_array = np.argmax(segmented_array, axis=0).astype(np.uint8)
        #segmented_array = segmented_array.transpose(1, 2, 0)

        segmented_array_list.append(segmented_array)

    """ Restore module. """
    segmented = extractor.restore(segmented_array_list)

    createParentPath(args.save_path)
    print("Saving image to {}".format(args.save_path))
    sitk.WriteImage(segmented, args.save_path, True)


if __name__ == '__main__':
    args = ParseArgs()
    main(args)
    
