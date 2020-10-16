from pytorch_lightning import Trainer
import pytorch_lightning
from pytorch_lightning.loggers import WandbLogger
 
from data import DMLDataModule, make_transform_inception_v3, FoodDataset, CarsDataset, dataset_names
from proxyNCA import DML
import wandb
classes_filename = "/home/ubuntu/few-shot-metric-learning/src/UMPC-G20.txt"
food_classes = FoodDataset.load_classes(classes_filename)
project_name="ProxyNCA-v2"
batch_size=32
gradient_clip_val=2.0
wandb.init(name=dataset_names[FoodDataset], project=project_name,reinit=True)

dm = DMLDataModule(
    name=dataset_names[FoodDataset],
    DataSetType=FoodDataset,
    root="/mnt/vol_b/images",
    # root="/mnt/vol_b/UPMC_Food101/images/",
    classes=food_classes[::2],
    eval_classes=food_classes[1::2],
    
    # name="Cars196",
    # DataSetType=CarsDataset,
    # root="/mnt/vol_b/cars",
    # classes=range(0, 98),
    # eval_classes=range(98, 196),

    batch_size=batch_size,
    train_transform=make_transform_inception_v3(augment=True),
    eval_transform=make_transform_inception_v3(augment=False)
)

wandb_logger = WandbLogger(name=dm.name, project=project_name, save_dir="/mnt/vol_b/models/few-shot")
dm.setup(project_name=project_name)
load_from_pth = "/home/ubuntu/few-shot-metric-learning/lightning_logs/version_11/checkpoints/epoch=0.ckpt"
if load_from_pth and  False:
    model = DML.load_from_checkpoint(load_from_pth)
else:

    model = DML(
        val_dataset=dm.val_dataset,
        num_classes=dm.num_classes,
        pooling="max",
        pretrained=True,
        lr_backbone=0.01,
        weight_decay_backbone=0.0,
        lr_embedding=0.001,
        weight_decay_embedding=0.0,
        lr=0.001,
        weight_decay_proxynca=0.0,
        dataloader=dm.train_dataloader(),
        scaling_x=3.0,
        scaling_p=3.0,
        smoothing_const=0.1,
        vis_dim=(2,3),
        batch_size=batch_size
    )
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
lr_logger = LearningRateMonitor(logging_interval='step')

trainer = Trainer(max_epochs=50, gpus=1,
                     logger=wandb_logger,
                    #  logger=True,
                    
                    #  fast_dev_run=True,
                    #  val_check_interval=0.1,
                    #  limit_val_batches=0.0,
                    gradient_clip_val=gradient_clip_val,
                    # auto_lr_find="lr",
                    #  overfit_batches=1,
                    # weights_summary='full',
                    # track_grad_norm=2,
                    callbacks=[lr_logger]
                     )
#  Start training
trainer.fit(model, datamodule=dm)

