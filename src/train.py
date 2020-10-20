from pytorch_lightning import Trainer
import pytorch_lightning
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf 
from data import DMLDataModule, make_transform_inception_v3, FoodDataset, CarsDataset, dataset_names
from proxyNCA import DML
from ast import literal_eval
import wandb
conf = OmegaConf.load('src/config.yml')
if conf.dataset.name in ["UMPC-G20", "UMPC_Food101"]:
    classes_filename = conf.dataset.classes_filename
    food_classes = FoodDataset.load_classes(classes_filename)
    classes=food_classes[::2]
    eval_classes=food_classes[1::2]
    DataSetType=FoodDataset

elif conf.dataset.name in ["Cars196"]:
    name="Cars196",
    DataSetType=CarsDataset
    root="/mnt/vol_b/cars"
    classes=range(0, 98)
    eval_classes=range(98, 196)
else:
    raise NotImplementedError(f"Dataset {conf.dataset.name} is not supported!")
breakpoint()
dm = DMLDataModule(
    name=DataSetType.name,
    DataSetType=DataSetType,
    root=conf.dataset.root,
    classes=classes,
    eval_classes=eval_classes,
    

    batch_size=conf.model.batch_size,
    train_transform=make_transform_inception_v3(augment=True),
    eval_transform=make_transform_inception_v3(augment=False)
)

wandb_logger = WandbLogger(name=dm.name, project=conf.experiment.name, save_dir="/mnt/vol_b/models/few-shot")
dm.setup(project_name=conf.experiment.name)


model = DML(
    val_dataset=dm.val_dataset,
    num_classes=dm.num_classes,
    pooling=conf.model.pooling,
    pretrained=conf.model.pretrained,
    lr_backbone=conf.model.lr_backbone,
    weight_decay_backbone=conf.model.weight_decay_backbone,
    lr_embedding=conf.model.lr_embedding,
    weight_decay_embedding=conf.model.weight_decay_embedding,
    lr=conf.model.lr,
    weight_decay_proxynca=conf.model.weight_decay_proxynca,
    dataloader=dm.train_dataloader(),
    scaling_x=conf.model.scaling_x,
    scaling_p=conf.model.scaling_p,
    smoothing_const=conf.model.smoothing_const,
    batch_size=conf.model.batch_size,

    vis_dim=literal_eval(conf.model.vis_dim),
)


trainer = Trainer(max_epochs=conf.experiment.max_epochs, gpus=conf.experiment.gpus,
                     logger=wandb_logger,
                    gradient_clip_val=conf.model.gradient_clip_val,                     )
#  Start training
trainer.fit(model, datamodule=dm)

