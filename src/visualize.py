#%%
%load_ext autoreload
%autoreload 2

from data import CarsDataModule


dm = CarsDataModule(
    root="/mnt/vol_b/cars",
    classes=range(0, 98),
    test_classes=range(98, 196),
    batch_size=32,
)
dm.setup()
# %%
dm.show_batch(dm.train_dataloader())
# %%
x, y, _ = dm.train_dataset[1000]
dm.show_sample(x, y,  invert=False)
# %%
x, y, z = next(iter(dm.test_dataloader()))

# %%
y
# %%
from torchvision.transforms import ToPILImage
ToPILImage()(x[1]).convert("L")
# %%
