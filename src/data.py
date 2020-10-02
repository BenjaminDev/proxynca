import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import dataset
import scipy.io
from typing import Union, List
import os
import torch
import torchvision
import numpy as np
import PIL.Image
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

cars_mean = (0.485, 0.456, 0.406)
cars_std = (0.229, 0.224, 0.225)

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, root, classes, transform=None):
        self.classes = classes
        self.root = root
        self.transform = transform
        self.ys, self.im_paths, self.I = [], [], []

    def nb_classes(self):
        assert set(self.ys) == set(self.classes)
        return len(self.classes)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        im = PIL.Image.open(self.im_paths[index])
        # convert gray to rgb
        if len(list(im.split())) == 1:
            im = im.convert("RGB")
        if self.transform is not None:
            im = self.transform(im)
        return im, self.ys[index], index

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.im_paths = [self.im_paths[i] for i in I]


class CarsDataset(torch.utils.data.Dataset):
    def __init__(self, root, classes, transform=None):
        super().__init__()
        self.classes = classes
        self.root = root
        self.transform = transform
        self.ys, self.im_paths, self.I = [], [], []

        annos_fn = "cars_annos.mat"
        cars = scipy.io.loadmat(os.path.join(root, annos_fn))
        self.cars = cars
        ys = [int(a[5][0] - 1) for a in cars["annotations"][0]]
        im_paths = [a[0][0] for a in cars["annotations"][0]]
        index = 0
        for im_path, y in zip(im_paths, ys):
            if y in classes:  # choose only specified classes
                self.im_paths.append(os.path.join(root, im_path))
                self.ys.append(y)
                self.I += [index]
                index += 1

    def nb_classes(self):
        assert set(self.ys) == set(self.classes)
        return len(self.classes)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        im = PIL.Image.open(self.im_paths[index])
        # convert gray to rgb
        if len(list(im.split())) == 1:
            im = im.convert("RGB")
        if self.transform is not None:
            im = self.transform(im)
        return im, self.ys[index], index

    def get_label(self, index):
        return self.ys[index]

    # def set_subset(self, I):
    #     self.ys = [self.ys[i] for i in I]
    #     self.I = [self.I[i] for i in I]
    #     self.im_paths = [self.im_paths[i] for i in I]


def std_per_channel(images):
    images = torch.stack(images, dim=0)
    return images.view(3, -1).std(dim=1)


def mean_per_channel(images):
    images = torch.stack(images, dim=0)
    return images.view(3, -1).mean(dim=1)


class Identity:  # used for skipping transforms
    def __call__(self, im):
        return im


class RGBToBGR:
    def __call__(self, im):
        assert im.mode == "RGB"
        r, g, b = [im.getchannel(i) for i in range(3)]
        # RGB mode also for BGR, `3x8-bit pixels, true color`, see PIL doc
        im = PIL.Image.merge("RGB", [b, g, r])
        return im


class ScaleIntensities:
    def __init__(self, in_range, out_range):
        """ Scales intensities. For example [-1, 1] -> [0, 255]."""
        self.in_range = in_range
        self.out_range = out_range

    def __oldcall__(self, tensor):
        tensor.mul_(255)
        return tensor

    def __call__(self, tensor):
        tensor = (tensor - self.in_range[0]) / (self.in_range[1] - self.in_range[0]) * (
            self.out_range[1] - self.out_range[0]
        ) + self.out_range[0]
        return tensor

def make_transform(mean=cars_mean, std=cars_std):
    normalize = transforms.Normalize(mean=mean,
                                 std=std)
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        #transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    

def make_transform_old(
    sz_resize=256,
    sz_crop=227,
    mean=[104, 117, 128],
    std=[1, 1, 1],
    rgb_to_bgr=True,
    is_train=True,
    intensity_scale=None,
):
    return transforms.Compose(
        [
            RGBToBGR() if rgb_to_bgr else Identity(),
            transforms.RandomResizedCrop(sz_crop) if is_train else Identity(),
            transforms.Resize(sz_resize) if not is_train else Identity(),
            transforms.CenterCrop(sz_crop) if not is_train else Identity(),
            transforms.RandomHorizontalFlip() if is_train else Identity(),
            transforms.ToTensor(),
            ScaleIntensities(*intensity_scale)
            if intensity_scale is not None
            else Identity(),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )


class CarsDataModule(pl.LightningDataModule):
    def __init__(
        self, root, classes, test_classes, transform=None, batch_size: int = 32
    ) -> None:
        super().__init__()
        self.transform = transform if transform else make_transform()
        self.root = root
        self.test_classes = test_classes
        self.classes = classes
        self.batch_size = batch_size
        self.num_classes = len(classes)

    def setup(self):
        self.train_dataset = CarsDataset(
            root=self.root, classes=self.classes, transform=self.transform
        )

        self.val_dataset = CarsDataset(
            root=self.root, classes=self.test_classes, transform=self.transform
        )

        self.test_dataset = CarsDataset(
            root=self.root, classes=self.test_classes, transform=self.transform
        )


    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, 
            num_workers=os.cpu_count(), shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:

        return DataLoader(
            dataset=self.val_dataset, batch_size=self.batch_size, 
            num_workers=os.cpu_count(), 
            shuffle=False,
            pin_memory=True
        )

    
    def test_dataloader(self):
        
        return DataLoader(
            dataset=self.test_dataset, batch_size=self.batch_size, num_workers=os.cpu_count(), shuffle=False,
            pin_memory=True
        )

    def show_batch(self, dl):
        for images, labels, _ in dl:
            fig, ax = plt.subplots(figsize=(16, 8))
            ax.set_xticks([]); ax.set_yticks([])
            # ax.set_xlabel("test")
            data = self.denorm(images).clamp(0,1)
            ax.imshow(make_grid(data, nrow=16).permute(1, 2, 0))
            break

    def show_sample(self, img, target, invert=True):
        img = self.denorm(img).clamp(0,1)
        if invert:
            plt.set_xlabel("help")
            plt.imshow(1 - img.permute((1, 2, 0)))
        else:
            plt.title(f"{target}")
            plt.imshow(img.permute(1, 2, 0))
    
    @staticmethod
    def denorm(img,mean=cars_mean, std=cars_std):
        return img*torch.Tensor(std).unsqueeze(1).unsqueeze(1)+torch.Tensor(mean).unsqueeze(1).unsqueeze(1)

class UPMCFood101DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (1, 28, 28)

    def prepare_data(self):
        # download
        pass
        # TODO: down load data if it's not available locally.
        # MNIST(self.data_dir, train=True, download=True)
        # MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        pass
        # Assign train/val datasets for use in dataloaders
        # if stage == 'fit' or stage is None:
        #     mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
        #     self._train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        #     # Optionally...
        #     # self.dims = tuple(self.mnist_train[0][0].shape)

        # # Assign test dataset for use in dataloader(s)
        # if stage == 'test' or stage is None:
        #     self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        # Optionally...
        # self.dims = tuple(self.mnist_test[0][0].shape)

    def train_dataloader(self):
        self.UPMCFood101_train = dataset
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self, batch_size=32)


# import hydra
# from omegaconf import DictConfig, OmegaConf

# @hydra.main(config_name="config.yml")
def main():

    dm = CarsDataModule(
        root="/mnt/c/Users/benja/workspace/data/cars", classes=range(0, 98)
    )
    breakpoint()


if __name__ == "__main__":
    main()

# dl_ev = torch.utils.data.DataLoader(
#     dataset.load(
#         name = args.dataset,
#         root = config['dataset'][args.dataset]['root'],
#         classes = config['dataset'][args.dataset]['classes']['eval'],
#         transform = dataset.utils.make_transform(
#             **config['transform_parameters'],
#             is_train = False
#         )
#     ),
#     batch_size = args.sz_batch,
#     shuffle = False,
#     num_workers = args.nb_workers,
#     pin_memory = True
# )
