import torch.utils.data as data
from .utils import separateData
from pathlib import Path

class UNetDataset(data.Dataset):
    def __init__(self, image_path_list, label_path, phase="train", criteria=None, transform=None):
        self.transform = transform
        self.phase = phase

        self.data_list = separateData(image_path_list, label_path, criteria, phase)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        path = self.data_list[index]
        image_array_list, label_array = self.transform(self.phase, *path)

        return image_array_list, label_array


if __name__ == '__main__':
    criteria = {
        "train" :["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"],
        "val" : ["20", "21", "22", "23", "24", "25", "26", "27", "28", "29"]
    }

    image_path_list = ["/mnt/data/patch/Abdomen/with_pad/concat_image/fold1/layer_1", "/mnt/data/patch/Abdomen/with_pad/concat_image/fold1/layer_2", "/mnt/data/patch/Abdomen/with_pad/128-128-8-1/image"]
    label_path = "/mnt/data/patch/Abdomen/with_pad/512-512-32-1/mask/image"

    from transform import UNetTransform
    dataset = UNetDataset(image_path_list, label_path, "train", criteria, transform = UNetTransform())

    print(dataset.__len__())
    x = dataset.__getitem__(20)
    print(x[0][0].shape, x[0][1].shape, x[0][2].shape, x[1].shape)



