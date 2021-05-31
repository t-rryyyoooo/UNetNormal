import pytorch_lightning as pl
import argparse
from pathlib import Path
import sys
sys.path.append("..")
from pytorch_lightning.loggers import TensorBoardLogger
from utils.utils import setSeed, printArgs
from model.UNet_no_pad_with_nonmask.system import UNetSystem as System

def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_mask_path", help="/home/vmlab/Desktop/data/patch/Abdomen/28-44-44/image")
    parser.add_argument("dataset_nonmask_path", help="/home/vmlab/Desktop/data/patch/Abdomen/28-44-44/image")
    parser.add_argument("log_path", help="/home/vmlab/Desktop/data/modelweight/Abdomen/28-44-44/mask")
    parser.add_argument("--train_list", help="00 01", nargs="*", default= "00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19")
    parser.add_argument("--val_list", help="20 21", nargs="*", default="20 21 22 23 24 25 26 27 28 29")
    parser.add_argument("--train_mask_nonmask_rate", nargs=2, default=[1.0, 0.1], type=float)
    parser.add_argument("--val_mask_nonmask_rate", nargs=2, default=[1.0, 1.0], type=float)
    parser.add_argument("--in_channel", help="Input channlel", type=int, default=1)
    parser.add_argument("--num_class", help="The number of classes.", type=int, default=14)
    parser.add_argument("--lr", help="Default 0.001", type=float, default=0.001)
    parser.add_argument("--batch_size", help="Default 6", type=int, default=6)
    parser.add_argument("--dropout", help="Default 6", type=float, default=0.5)
    parser.add_argument("--num_workers", help="Default 6.", type=int, default=6)
    parser.add_argument("--epoch", help="Default 50.", type=int, default=50)
    parser.add_argument("--gpu_ids", help="Default 0.", type=int, default=0, nargs="*")

    args = parser.parse_args()

    return args

def main(args):
    setSeed()
    printArgs(args)

    criteria = {
            "train" : args.train_list, 
            "val"   : args.val_list
            }

    rate = {
            "train" : {
                "mask"    : args.train_mask_nonmask_rate[0], 
                "nonmask" : args.train_mask_nonmask_rate[1]
                },
            "val" : {
                "mask"    : args.val_mask_nonmask_rate[0], 
                "nonmask" : args.val_mask_nonmask_rate[1]
                }
            }

    system = System(
                dataset_mask_path    = args.dataset_mask_path,
                dataset_nonmask_path = args.dataset_nonmask_path,
                log_path             = args.log_path,
                criteria             = criteria,
                rate                 = rate,
                in_channel           = args.in_channel,
                num_class            = args.num_class,
                learning_rate        = args.lr,
                batch_size           = args.batch_size,
                dropout              = args.dropout,
                num_workers          = args.num_workers
                )

    logger = TensorBoardLogger(args.log_path)

    trainer = pl.Trainer(
                num_sanity_val_steps = 0, 
                max_epochs           = args.epoch,
                checkpoint_callback  = None, 
                logger               = logger,
                gpus                 = args.gpu_ids,
                )
 
    trainer.fit(system)

if __name__ == "__main__":
    args = parseArgs()
    main(args)
