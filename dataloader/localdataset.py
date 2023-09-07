import glob
from PIL import Image
from functools import partial

from torch import nn
from torchvision import transforms
from torch.utils import data as data

from myutils.img_util import convert_image_to_fn

def exists(x):
    return x is not None

class LocalLQImageDataset(data.Dataset):
    def __init__(self, lq_folder, image_size=256, random_flip=True, center_crop=False, convert_image_to='RGB'):
        super(LocalLQImageDataset, self).__init__()

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to, image_size) if exists(convert_image_to) else nn.Identity()
        self.img_preproc = transforms.Compose([
            #transforms.Resize(image_size),
            transforms.Lambda(maybe_convert_fn),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.CenterCrop(image_size) if center_crop else transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.paths = sorted(glob.glob(f'{lq_folder}/*.*g'))[:100]

    def __getitem__(self, index):
        example = dict()

        # load lq image
        lq_path = self.paths[index]
        img_lq = Image.open(lq_path).convert('RGB')

        example["pixel_values"] = self.img_preproc(img_lq)
        
        return example

    def __len__(self):
        return len(self.paths)