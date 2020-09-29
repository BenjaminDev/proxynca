
from torch import Tensor
from src.data import CarsDataModule
def test_cars_datamodule():
    dm = CarsDataModule(root="/mnt/c/Users/benja/workspace/data/cars", 
                        classes=range(0, 98))
    t_dl = dm.train_dataloader()                        
    _, x, y = next(iter(t_dl))
    assert isinstance(x, Tensor)