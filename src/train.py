from re import escape
from pytorch_lightning import Trainer
import pytorch_lightning
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import lr_finder
from data import CarsDataModule
from model import get_inception_v3_model, get_inception_v2_model
from task import DML, ProxyNCA

dm = CarsDataModule(
    root="/mnt/vol_b/cars",
    classes=range(0, 98),
    test_classes=range(98, 196),
    batch_size=64,
)
# dm.prepare_data()
# dm.setup('fit')
load_from_pth = "/home/ubuntu/few-shot-metric-learning/lightning_logs/version_11/checkpoints/epoch=0.ckpt"
if load_from_pth and  False:
    model = DML.load_from_checkpoint(load_from_pth)
else:   
    model = DML(
        model=get_inception_v2_model(sz_embedding=64),
        criterion=ProxyNCA(nb_classes=dm.num_classes, sz_embedding=64),
        lr_backbone=0.001,
        weight_decay_backbone=0.0,
        lr_embedding=0.001,
        weight_decay_embedding=0.0,
        lr_proxynca=0.001,
        weight_decay_proxynca=0.0,
    )

wandb_logger = WandbLogger(name='Adam',project='few-shot', save_dir="/mnt/vol_b/models/few-shot")
trainer = Trainer(max_epochs=100, gpus=1,
                     logger=wandb_logger,
                     fast_dev_run=True,
                     )

trainer.fit(model, datamodule=dm)
trainer.test(model, datamodule=dm)
