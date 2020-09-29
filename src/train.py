from pytorch_lightning import Trainer
from data import CarsDataModule
from model import get_model
from task import DML, ProxyNCA

dm = CarsDataModule(
    root="/mnt/c/Users/benja/workspace/data/cars",
    classes=range(0, 98),
    test_classes=range(98, 196),
    batch_size=32,
)
# dm.prepare_data()
# dm.setup('fit')

model = DML(
    model=get_model(sz_embedding=64),
    criterion=ProxyNCA(nb_classes=10, sz_embedding=64),
    lr_backbone=0.001,
    weight_decay_backbone=0.0,
    lr_embedding=0.001,
    weight_decay_embedding=0.0,
    lr_proxynca=0.001,
    weight_decay_proxynca=0.0,
)

trainer = Trainer(max_epochs=2, gpus=None, logger=True, fast_dev_run=False)

trainer.fit(model, datamodule=dm)
trainer.test(model, datamodule=dm)
# trainer.test(datamodule=dm
