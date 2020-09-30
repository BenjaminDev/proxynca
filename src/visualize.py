#%%
from .data import CarsDataModule


dm = CarsDataModule(
    root="/mnt/c/Users/benja/workspace/data/cars",
    classes=range(0, 98),
    test_classes=range(98, 196),
    batch_size=32,
)
# %%
x, y, z = next(iter(dm.test_dataloader()))

# %%
y
# %%
from torchvision.transforms import ToPILImage
ToPILImage()(x[1]).convert("L")
# %%
