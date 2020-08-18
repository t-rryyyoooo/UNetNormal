from .preprocessing import Compose, LoadNumpys

class UNetTransform():
    def __init__(self, translate_range=0., rotate_range=0., shear_range=0., scale_range=0.):
        self.translate_range = translate_range
        self.rotate_range = rotate_range
        self.shear_range = shear_range
        self.scale_range = scale_range

        self.transforms = {
                "train" : Compose([
                        LoadNumpys()
                    ]), 

                "val" : Compose([
                        LoadNumpys()
                    ])
                }

    def __call__(self, phase, image_list, label):

        return self.transforms[phase](image_list, label)

if __name__ == "__main__":

    path = (['/mnt/data/patch/Abdomen/with_pad/concat_image/fold1/layer_1/case_02/image_005.npy', '/mnt/data/patch/Abdomen/with_pad/concat_image/fold1/layer_2/case_02/image_005.npy', '/mnt/data/patch/Abdomen/with_pad/128-128-8-1/image/case_02/image_005.npy'], '/mnt/data/patch/Abdomen/with_pad/512-512-32-1/mask/image/case_02/label_005.npy')
    ut = UNetTransform()
    x = ut("train", *path)
    print(x[0][0].shape, x[0][1].shape, x[0][2].shape, x[1].shape)

