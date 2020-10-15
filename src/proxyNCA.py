from random import randint
import random
from typing import Any, Dict, Tuple, Union
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
# import plotly.express as px
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from torch import optim
from sklearn.decomposition import PCA
from torchvision import models
from torch.nn import Parameter
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
import wandb
from evaluation import (assign_by_euclidian_at_k,
                        calc_normalized_mutual_information, calc_recall_at_k,
                        cluster_by_kmeans)

colors_by_name = ["aliceblue", "antiquewhite", "aqua", "aquamarine", "azure",
            "beige", "bisque", "black", "blanchedalmond", "blue",
            "blueviolet", "brown", "burlywood", "cadetblue",
            "chartreuse", "chocolate", "coral", "cornflowerblue",
            "cornsilk", "crimson", "cyan", "darkblue", "darkcyan",
            "darkgoldenrod", "darkgray", "darkgrey", "darkgreen",
            "darkkhaki", "darkmagenta", "darkolivegreen", "darkorange",
            "darkorchid", "darkred", "darksalmon", "darkseagreen",
            "darkslateblue", "darkslategray", "darkslategrey",
            "darkturquoise", "darkviolet", "deeppink", "deepskyblue",
            "dimgray", "dimgrey", "dodgerblue", "firebrick",
            "floralwhite", "forestgreen", "fuchsia", "gainsboro",
            "ghostwhite", "gold", "goldenrod", "gray", "grey", "green",
            "greenyellow", "honeydew", "hotpink", "indianred", "indigo",
            "ivory", "khaki", "lavender", "lavenderblush", "lawngreen",
            "lemonchiffon", "lightblue", "lightcoral", "lightcyan",
            "lightgoldenrodyellow", "lightgray", "lightgrey",
            "lightgreen", "lightpink", "lightsalmon", "lightseagreen",
            "lightskyblue", "lightslategray", "lightslategrey",
            "lightsteelblue", "lightyellow", "lime", "limegreen",
            "linen","magenta", "maroon", "mediumaquamarine",
            "mediumblue", "mediumorchid", "mediumpurple",
            "mediumseagreen", "mediumslateblue", "mediumspringgreen",
            "mediumturquoise", "mediumvioletred", "midnightblue",
            "mintcream", "mistyrose", "moccasin", "navajowhite", "navy",
            "oldlace", "olive", "olivedrab", "orange", "orangered",
            "orchid", "palegoldenrod", "palegreen", "paleturquoise",
            "palevioletred", 'papayawhip', "peachpuff", 'peru', "pink",
            'plum', "powderblue", 'purple', "red", 'rosybrown',
            "royalblue", "rebeccapurple", "saddlebrown", "salmon",
            "sandybrown", "seagreen", "seashell", "sienna", "silver",
            "skyblue", "slateblue", "slategray", "slategrey", "snow",
            "springgreen", "steelblue", 'tan', "teal", "thistle", "tomato",
            "turquoise", "violet", "wheat", "white", "whitesmoke",
            "yellow", "yellowgreen"]

random.seed(30)
colors_by_name = [r] 

def binarize_and_smooth_labels(T, num_classes, smoothing_const=0.1):
    # REF: https://github.com/dichotomies/proxy-nca
    # Optional: BNInception uses label smoothing, apply it for retraining also
    # "Rethinking the Inception Architecture for Computer Vision", p. 6
    import sklearn.preprocessing

    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(T, classes=range(0, num_classes))
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (num_classes - 1)
    T = torch.FloatTensor(T)
    return T


class DML(pl.LightningModule):
    """Distance Metric Learning"""
    def __init__(self, *,val_dataset:Dataset, num_classes:int, sz_embedding:int=64, backbone:str="inception_v3",**kwargs) -> None:
        """DML module

        Args:
            val_dataset (Dataset): dataset holding the validation data. 
            num_classes (int): Number of classes.
            sz_embedding (int, optional): Size of the embedding to use. Defaults to 64.
            backbone (str, optional): Backbone architecture. Defaults to "inception_v3".

        Raises:
            NotImplementedError: On backbone architecture not supported.
            NotImplementedError: On pooling type not supported.
        """
        super().__init__()
        self.save_hyperparameters()
        self.val_dataset=val_dataset
        # Backbone
        if backbone == "inception_v3":
            inception = models.inception_v3(pretrained=self.hparams.pretrained or True)
            self.transform_input = True
            self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
            self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
            
            self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
            self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

            self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
            self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
            self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
            self.Mixed_5b = inception.Mixed_5b
            self.Mixed_5c = inception.Mixed_5c
            self.Mixed_5d = inception.Mixed_5d
            self.Mixed_6a = inception.Mixed_6a    
            self.Mixed_6b = inception.Mixed_6b
            self.Mixed_6c = inception.Mixed_6c
            self.Mixed_6d = inception.Mixed_6d
            self.Mixed_6e = inception.Mixed_6e
            self.Mixed_7a = inception.Mixed_7a
            self.Mixed_7b = inception.Mixed_7b
            self.Mixed_7c = inception.Mixed_7c
            
            self.in_features=2048
        else: raise NotImplementedError(f"backbone {backbone} is not supported!")
        # Global Pooling
        if self.hparams.pooling=="avg":
            self.global_pool = torch.nn.AvgPool2d (8, stride=1, padding=0, ceil_mode=True, count_include_pad=True)
        elif self.hparams.pooling=="max":
            self.global_pool = torch.nn.MaxPool2d (8, stride=1, padding=0)
        else:
            raise NotImplementedError(f"pooling {self.hparams.pooling} not supported!")
        # Embedding    
        self.embedding_layer = torch.nn.Linear(in_features=self.in_features, out_features=sz_embedding)
        # Proxies
        self.proxies = Parameter(torch.randn(num_classes, sz_embedding) / 8, requires_grad=True)
        self.proxies.register_hook(lambda grad: self.logger.experiment.log({"proxy_grad":grad.cpu()})) 
    
    def _transform_input(self, x):
        """Fixes the difference between pytorch and tensorflow normalizing conventions"""
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def forward(self, x):
        """REF: torchvision.models"""
        x = self._transform_input(x)
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.embedding_layer(x)
        return x

    def compute_loss(self, images, target, include_embeddings=False)->Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        X = self(images)
        P = F.normalize(self.proxies, p = 2, dim = -1) * self.hparams.scaling_p
        X = F.normalize(X, p = 2, dim = -1) * self.hparams.scaling_x
        D = torch.cdist(X, P) ** 2
        T = binarize_and_smooth_labels(target, len(P), self.hparams.smoothing_const).to(X.device)
        
        # note that compared to proxy nca, positive included in denominator
        loss = torch.sum(-T * F.log_softmax(-D, -1), -1)
        if include_embeddings:
            return loss.mean(), X
        return loss.mean()

    def training_step(self, batch, batch_idx)-> Dict[str, Any]:
        """Run a batch from the train dataset through the model"""
        images, target, _ = batch
        
        loss, Xs = self.compute_loss(images, target,include_embeddings=True)
        self.log_dict({"train_loss": loss},prog_bar=True, on_step=True, on_epoch=True)
        return {"loss":loss, "Xs":Xs, "Ts":target }

    def training_epoch_end(self, outputs):
        """"""
        # Since validation set samples are iid I prefer looking at a histogram of validation losses.
        wandb.log({f"train_loss_hist": wandb.Histogram([[h["loss"].cpu() for h in outputs]])})   

    def validation_step(self, batch, batch_idx) -> Dict[str, Any]:
        """Run a batch from the validation dataset through the model"""
        images, target, index = batch

        val_loss, Xs = self.compute_loss(images, target, include_embeddings=True)

        self.log_dict({"val_loss":val_loss}, prog_bar=True, on_step=False, on_epoch=True)

        return {"Xs":Xs, "Ts":target, "index":index, "val_loss":val_loss.item()}

    def validation_epoch_end(self, outputs:Dict[str, Any])->None:
        """Compute metrics on the full validation set.

        Args:
            outputs (Dict[str, Any]): Dict of values collected over each batch put through model.eval()(..) 
        """

        val_Xs = torch.cat([h["Xs"] for h in outputs])
        val_Ts = torch.cat([h["Ts"] for h in outputs])
        val_indexes = torch.cat([h["index"] for h in outputs])
        Y = assign_by_euclidian_at_k(val_Xs.cpu(), val_Ts.cpu(), 8) 
        Y = torch.from_numpy(Y)
        
        # Return early when PL is running the sanity check.
        if self.trainer.running_sanity_check:
            return

        # Compute and Log R@k
        recall = []
        logs = {}
        for k in [1, 2, 4, 8]:
            r_at_k = 100*calc_recall_at_k(val_Ts.cpu(), Y, k)
            recall.append(r_at_k)
            logs[f"val_R@{k}"] = r_at_k
        self.log_dict(logs)
        
        # Compute and log NMI
        nmi = 100*calc_normalized_mutual_information(
            val_Ts.cpu(),
            cluster_by_kmeans(
                val_Xs.cpu(), self.hparams.num_classes
            )
        )
        self.log_dict({"NMI":nmi})
        
        # Inspect the embedding space.        
        pca = PCA(3)  
        projected = pca.fit_transform(val_Xs.cpu())
        # fig = go.Figure(data=go.Scatter(x=projected[:, 0],
        #                         y= projected[:, 1],
        #                         mode='markers',
        #                         marker_color=[colors_by_name[o%len(colors_by_name)] for o in range(0,self.hparams.num_classes)],
        #                         text=[self.val_dataset.get_label_description(o) for o in Y[:,0]])) # hover text goes here
        fig = go.Figure(data=[go.Scatter3d(x=projected[:, 0], y=projected[:, 1], z=projected[:, 2],
                                   marker_color=[colors_by_name[o%len(colors_by_name)] for o in range(0,len(Y[:,0]))],
                                   text=[self.val_dataset.get_label_description(o) for o in Y[:,0]], 
                                   mode='markers')])
        wandb.log({"Embedding of Validation Dataset": fig})

        wandb.sklearn.plot_confusion_matrix(val_Ts.cpu().numpy(), Y[:,0], labels=self.val_dataset.classes)
            
        # Project the proxies onto the same 2D space
        proxies = pca.transform(self.proxies.detach().cpu())
        # fig = go.Figure(data=go.Scatter(x=proxies[:, 0],
        #                         y= proxies[:, 1],
        #                         mode='markers',
        #                         pythoncolor=[colors_by_name[o%len(colors_by_name)] for o in range(0,self.hparams.num_classes)],
        #                         text=[self.val_dataset.get_label_description(o) for o in range(0,self.hparams.num_classes)])) # hover text goes here
        fig = go.Figure(data=[go.Scatter3d(x=proxies[:,0], y=proxies[:,1], z=proxies[:,2],
                                   mode='markers',
                                   marker_color=[colors_by_name[o%len(colors_by_name)] for o in range(0,self.hparams.num_classes)],
                                   text=[self.val_dataset.get_label_description(o) for o in range(0,self.hparams.num_classes)]) # hover text goes here
                                   ])
        wandb.log({"Embedding of Proxies (on validation data)": fig})

       
        # Log a query and top 4 selction
        image_dict={}
        top_k_indices = torch.cdist(val_Xs,val_Xs).topk(5, largest=False).indices
        max_idx = len(top_k_indices) -1
        for i, example_result in enumerate(top_k_indices[[randint(0,max_idx) for _ in range(0,5)]]):
             
            image_dict[f"global step {self.global_step} example: {i}"] = [wandb.Image(Image.open(self.val_dataset.im_paths[val_indexes[example_result[0]]]), caption=f"query: {self.val_dataset.get_label_description(self.val_dataset.get_label(val_indexes[example_result[0]]))}") ]
            image_dict[f"global step {self.global_step} example: {i}"].extend([wandb.Image(Image.open(self.val_dataset.im_paths[val_indexes[idx]]), caption=f"retrival:({rank}) {self.val_dataset.get_label_description(self.val_dataset.get_label(val_indexes[idx]))}") for rank, idx in enumerate(example_result[1:])])
        self.logger.experiment.log(image_dict) 
        
        # Since validation set samples are iid I prefer looking at a histogram of valitation losses.
        wandb.log({f"val_loss_hist": wandb.Histogram([[h["val_loss"] for h in outputs]])})    

    def configure_optimizers(self):
        """Setup the optimizer configuration."""
        parameters = [p for p in self.parameters()]
        backbone_parameters = parameters[:-2]
        embedding_parameters = parameters[-2:-1]
        proxy_parameters = parameters[-1:]

        
        optimizer = optim.Adam(
            [
                {
                    "params": backbone_parameters,
                    "lr": self.hparams.lr_backbone,
                    "eps": 1.0,
                    "weight_decay": self.hparams.weight_decay_backbone,
                },
                {
                    "params": embedding_parameters,
                    "lr": self.hparams.lr_embedding,
                    "eps": 1.0,
                    "weight_decay": self.hparams.weight_decay_embedding,
                },
                {
                    "params": proxy_parameters,
                    "lr": self.hparams.lr,
                    "eps": 1.0,
                    "weight_decay": self.hparams.weight_decay_proxynca,
                },
            ],
        )

        # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.94)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.lr)

        return [optimizer], [scheduler]
