import os
from pathlib import Path
from typing import Callable, List, Optional, Union

import kornia
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pytorch_lightning as pl
import scipy.io
import torch
import torchvision.transforms as T
from fastcore.utils import parallel
from fastprogress import progress_bar
from torch.utils.data import DataLoader, dataset
from torchvision import transforms
from torchvision.utils import make_grid
import wandb

cars_mean = (0.485, 0.456, 0.406)
cars_std = (0.229, 0.224, 0.225)

class DMLDataModule(pl.LightningDataModule):
    """A generic Distance Metric Learning datamodule.
    """
    def __init__(
        self,
        name:str,
        DataSetType,
        root:Union[Path, str],
        classes:List[Union[int, str]],
        eval_classes:List[Union[int, str]],
        train_transform:Callable,
        eval_transform:Callable,
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        self.name = name
        self.DataSetType = DataSetType
        self.root = root
        self.classes = [o for o in classes]
        self.eval_classes = eval_classes
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.batch_size = batch_size
        self.num_classes = len(classes)

    def setup(self, project_name:str):
        self.train_dataset = self.DataSetType(
            root=self.root, classes=self.classes, transform=self.train_transform
        )

        self.val_dataset = self.DataSetType(
            root=self.root, classes=self.eval_classes, transform=self.eval_transform
        )

        def plot_class_distributions(ys, classes):
            import plotly.graph_objects as go
            labels, counts = zip(*Counter(ys).items())
            labels=[classes[o] for o in labels]
            fig = go.Figure([go.Bar(x=labels, y=counts)])
            return fig
        wandb.init(name=self.name, project=project_name, reinit=True)
        wandb.log({"Validation Class Distribution": plot_class_distributions(ys=self.val_dataset.ys, classes=self.val_dataset.classes)})
        wandb.log({"Train Class Distribution": plot_class_distributions(ys=self.train_dataset.ys, classes=self.train_dataset.classes)})

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:

        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            shuffle=False,
            pin_memory=True,
        )

    def show_batch(self, dl):
        for images, labels, _ in dl:
            fig, ax = plt.subplots(figsize=(16, 8))
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.set_xlabel("test")
            data = self.denorm(images).clamp(0, 1)
            ax.imshow(make_grid(data, nrow=16).permute(1, 2, 0))
            break

    def show_sample(self, img, target, invert=True):
        img = self.denorm(img).clamp(0, 1)
        if invert:
            plt.set_xlabel(self.classes[target])
            plt.imshow(1 - img.permute((1, 2, 0)))
        else:
            plt.title(f"{self.classes[target]}")
            plt.imshow(img.permute(1, 2, 0))

    @staticmethod
    def denorm(img, mean=cars_mean, std=cars_std):
        return img * torch.Tensor(std).unsqueeze(1).unsqueeze(1) + torch.Tensor(
            mean
        ).unsqueeze(1).unsqueeze(1)


class CarsDataset(torch.utils.data.Dataset):
    """
    Loads the cars196 dataset.
    To download:
    ```
    wget http://imagenet.stanford.edu/internal/car196/cars_annos.mat
    wget http://imagenet.stanford.edu/internal/car196/car_ims.tgz
    tar -xzvf car_ims.tgz
    ```
    """

    def __init__(
        self, root: str, classes: List[int], transform: Optional[Callable] = None
    ):
        """Holds the cars196 dataset
           REF: https://github.com/dichotomies/proxy-nca
        Args:
            root (Union[Path, str]): Paths to `cars` dirctory.
            classes (List[int]): List of valid labels range(0,196)
            transform (Optional[Callable], optional): transform to apply. Defaults to None.
        """
        super().__init__()
        self.classes = [o for o in classes]
        self.root = root
        self.transform = transform
        self.ys, self.im_paths, self.I = [], [], []
        annos_fn = "cars_annos.mat"
        cars = scipy.io.loadmat(os.path.join(root, annos_fn))
        self.class_descriptions = [o[0] for o in cars["class_names"][0]]
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

    def get_label_description(self, label_idx:int)->Union[str, int]:
        if self.class_descriptions:
            return self.class_descriptions[label_idx]
        else: return str(label_idx)

from collections import Counter
class FoodDataset(torch.utils.data.Dataset):
    """
    Dataset for [UMPC-G20](http://visiir.lip6.fr/)
    Note: This dataset is set to load the 20 categories of images.
    Remove the train and test folders as they are the Food101 dataset.
    """

    def __init__(
        self, root: Union[Path, str], classes: List[str], transform=Optional[Callable]
    ):
        """Dataset for [UMPC-G20](http://visiir.lip6.fr/)

        Args:
            root (Path): path to `images` folder.
            classes (List[str]): list of classes to load.
            transform ([Callable], optional): transform to apply. Defaults to None.
        """
        super().__init__()
        normalize_names = lambda x: x.replace("-", "_") 
        self.classes = [normalize_names(o) for o in classes]
        self.root = root
        self.transform = transform

        valid_image_paths = sorted([p for p in Path(self.root).glob(f"**/**/*.jpg")])

        self.im_paths = [p for p in valid_image_paths if normalize_names(p.parent.stem) in self.classes]
        self.ys = [self.classes.index(normalize_names(p.parent.stem)) for p in self.im_paths]
        print(f"Class Counts: {Counter([normalize_names(p.parent.stem) for p in self.im_paths])}")


    def nb_classes(self):
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

    def get_label(self, index:int)->int:
        return self.ys[index]
    
    def get_label_description(self, label_idx:int)->str:

        return str(self.classes[label_idx])

    @staticmethod
    def load_classes(filename):
        with open(filename, mode="r") as fp:
            return [l.strip() for l in fp.readlines()]
        

dataset_names={FoodDataset:"UMP-G20", CarsDataset:"Cars196"}

#################### Utils#########################
def verify_image(fn):
    "Confirm that `fn` can be opened. REF: fastai."
    try:
        im = PIL.Image.open(fn)
        im.draft(im.mode, (32, 32))
        im.load()
        return fn
    except:
        os.remove(fn)
        return None


def remove_broken_images(image_paths: List[Path]) -> List:
    """removes images that cannot be opened by PIL.

    Args:
        image_paths (List[Path]): List of `Path`s to images that are to be checked.

    Returns:
        List: List of `Path`s where all paths are to valid images.
    """
    return parallel(verify_image, image_paths, progress=progress_bar, threadpool=True)


def reduce_batch_of_one(image: torch.Tensor) -> torch.Tensor:
    """reduces a tensor along the batch dimension

    Args:
        image (torch.Tensor): batch of one reduce

    Returns:
        torch.Tensor: image
    """
    return image.squeeze(0)


def make_transform_inception_v3(augment=False)->torch.Tensor:
    """Transformation pipeline for loading data into the inception_v3 backbone.

    Args:
        augment (bool, optional): if set data augmentation will be applied. Defaults to False.

    Returns:
        [torch.Tensor]: returns image as tensor. 
    """

    base_transforms = [
        T.Resize(299),
        T.CenterCrop(299),
        T.ToTensor(),
    ]

    if augment:
        base_transforms =base_transforms + [torch.nn.Sequential(
            kornia.augmentation.RandomHorizontalFlip(),
            # kornia.augmentation.RandomGrayscale(p=0.001),
            kornia.augmentation.RandomRotation(degrees=180),
        ), reduce_batch_of_one]

    return transforms.Compose(
            base_transforms
            + [
                T.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            ]
        )
