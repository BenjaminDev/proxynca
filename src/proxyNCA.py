from logging import log
from numpy.core.shape_base import stack
import pytorch_lightning as pl
from pytorch_lightning import EvalResult, TrainResult
import torch
from torch import tensor
from torch._C import dtype
import torch.nn.functional as F
from torch.nn import Parameter
from torch import optim
from torch.optim import lr_scheduler
from PIL import Image
import wandb

from evaluation import assign_by_euclidian_at_k, calc_recall_at_k


def binarize_and_smooth_labels(T, nb_classes, smoothing_const=0.1):
    # Optional: BNInception uses label smoothing, apply it for retraining also
    # "Rethinking the Inception Architecture for Computer Vision", p. 6
    import sklearn.preprocessing

    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(T, classes=range(0, nb_classes))
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    T = torch.FloatTensor(T)  # .cuda()
    return T


class ProxyNCA(torch.nn.Module):
    def __init__(
        self, nb_classes, sz_embedding, smoothing_const=0.1, scaling_x=3, scaling_p=3
    ):
        super().__init__()
        # initialize proxies s.t. norm of each proxy ~1 through div by 8
        # i.e. proxies.norm(2, dim=1)) should be close to [1,1,...,1]
        # TODO: use norm instead of div 8, because of embedding size
        self.proxies = Parameter(torch.randn(nb_classes, sz_embedding) / 8)
        self.smoothing_const = smoothing_const
        self.scaling_x = scaling_x
        self.scaling_p = scaling_p

    def forward(self, X, T):
        P = F.normalize(self.proxies, p=2, dim=-1) * self.scaling_p
        X = F.normalize(X, p=2, dim=-1) * self.scaling_x
        D = torch.cdist(X, P) ** 2
        T = binarize_and_smooth_labels(T, len(P), self.smoothing_const)
        if T.device != D.device:
            T=T.to(D.device)
        # note that compared to proxy nca, positive included in denominator
        # breakpoint()
        loss = torch.sum(-T * F.log_softmax(-D, -1), -1)
        return loss.mean()
from torchvision import models
from torch.nn import Identity

def get_inception_v3_model(pretrained=True):

    inception_v3 = models.inception_v3(pretrained=pretrained)
    
    return inception_v3

class DML(pl.LightningModule):
    def __init__(self, *,nb_classes:int, sz_embedding:int=64, backbone:str="inception_v3", **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        # Backbone:
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
            # Auxilary head removed
            self.Mixed_7a = inception.Mixed_7a
            self.Mixed_7b = inception.Mixed_7b
            self.Mixed_7c = inception.Mixed_7c
            self.in_features=2048
        else: raise NotImplementedError(f"backbone {backbone} is not supported!")

        self.global_pool = torch.nn.AvgPool2d (8, stride=1, padding=0, ceil_mode=True, count_include_pad=True)
        self.embedding_layer = torch.nn.Linear(in_features=self.in_features, out_features=sz_embedding)
        self.proxies = Parameter(torch.randn(nb_classes, sz_embedding) / 8, requires_grad=True)
        self.proxies.register_hook(lambda grad: self.logger.experiment.log({"proxy_grad":grad.cpu()})) 
    def _transform_input(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def forward(self, x):
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

    def compute_loss(self, images, target, index=None,include_Xs_and_Ts=False):
        X = self(images)
        # breakpoint()
        P = F.normalize(self.proxies, p = 2, dim = -1) * self.hparams.scaling_p
        X = F.normalize(X, p = 2, dim = -1) * self.hparams.scaling_x
        D = torch.cdist(X, P) ** 2
        T = binarize_and_smooth_labels(target, len(P), self.hparams.smoothing_const).to(X.device)
        # note that compared to proxy nca, positive included in denominator
        loss = torch.sum(-T * F.log_softmax(-D, -1), -1)
        # breakpoint()
        if include_Xs_and_Ts and index is not None:
            breakpoint()
            return loss.mean(), X, target, index[D.topk(4, largest=False).indices]
        return loss.mean()
    def training_step(self, batch, batch_idx):
        
        images, target, index = batch
        # breakpoint()
        loss = self.compute_loss(images, target)
        # result = TrainResult(minimize=loss)
        self.log_dict({"train_loss": loss},prog_bar=True, on_step=True, on_epoch=False)
        # result.log_dict({"examples": [wandb.Image(Image.open("/mnt/vol_b/cars/car_ims/014839.jpg"), caption="Label")]})
        return loss
        # return {"loss": loss, "train_loss":loss.item() }


    def validation_step(self, batch, batch_idx) -> EvalResult:
        images, target, index = batch
        # breakpoint()
        val_loss, Xs, Ts = self.compute_loss(images, target, index=index, include_Xs_and_Ts=True)
        # breakpoint()
        # result = EvalResult(checkpoint_on=val_loss)
        self.log_dict({"val_loss":val_loss}, prog_bar=True, on_step=True, on_epoch=False)

        # X, T, *_ = self.predict_batchwise(batch)
        # result.hiddens = [X, T]

        return {"Xs":Xs, "Ts":Ts, "index":index}
    def validation_epoch_end(self, outputs):
        recall = []
        logs = {}
        # breakpoint()
        val_Xs = torch.cat([h["Xs"] for h in outputs])
        val_Ts = torch.cat([h["Ts"] for h in outputs])
        Y = assign_by_euclidian_at_k(val_Xs.cpu(), val_Ts.cpu(), 8) # min(8, len(batch)))
        Y = torch.from_numpy(Y)
        # breakpoint()
        for k in [1, 2, 4, 8]:
            r_at_k = 100*calc_recall_at_k(val_Ts.cpu(), Y, k)
            recall.append(r_at_k)
            logs[f"val_R@{k}"] = r_at_k  # f"{r_at_k:.3f}"
        self.log_dict(logs)
        images_and_labels = [self.dm.val_dataset[o["index"][0]] for o in outputs]
        image_dict={"Query": [wandb.Image(some_img, caption="query")]}
        # for img_pth in image_paths:

        self.logger.experiment.log({"proxy_grad":grad.cpu()}) 
    #     result = EvalResult()
    #     result.log_dict({"avg_val_loss": outputs["val_loss"].mean()})
    #     result.log_dict(logs)
    #     return result

    #     return torch.mean(outputs["val_loss"])

    # def validation_epoch_end(self, outputs)-> EvalResult:
    #     all_Xs = torch.empty(0).to(self.device)
    #     all_Ts = torch.empty(0).to(self.device)
    #     for X_T in outputs:
    #         X, T = X_T
    #         all_Xs = torch.cat([all_Xs, X])
    #         all_Ts = torch.cat([all_Ts, T])
    #     recall = []
    #     logs = {}
    #     breakpoint()
    #     Y = assign_by_euclidian_at_k(all_Xs, all_Ts, 8) # min(8, len(batch)))
    #     for k in [1, 2, 4, 8]:
    #         r_at_k = calc_recall_at_k(all_Ts, Y, k)
    #         recall.append(r_at_k)
    #         logs[f"Val_R@{k}"] = r_at_k  # f"{r_at_k:.3f}"
            

    #         # logging.info("R@{} : {:.3f}".format(k, 100 * r_at_k))
    #     # result.log_dict(logs)
    #     # print(logs)
    #     val_loss = self.criterion(all_Xs, all_Ts)
    #     result = EvalResult(checkpoint_on=val_loss)
    #     logs["val_loss"] = val_loss
    #     result.log_dict(logs)
    #     return result

    def predict_batchwise(self, batch):
        # list with N lists, where N = |{image, label, index}|
        A = [[] for i in range(len(batch))]
        # extract batches (A becomes list of samples)
        for i, J in enumerate(batch):
            # i = 0: sz_batch * images
            # i = 1: sz_batch * labels
            # i = 2: sz_batch * indices
            if i == 0:
                # move images to device of model (approximate device)
                # J = J.to(list(model.parameters())[0].device)
                # predict model output for image
                J = self.model(J)
            for j in J:
                A[i].append(j)
        return [torch.stack(A[i]) for i in range(len(A))]

    def test_step(self, batch, batch_idx) -> EvalResult:

        X, T, *_ = self.predict_batchwise(batch)
        if self.test_Xs.device != X.device: self.test_Xs = self.test_Xs.to(X.device)
        if self.test_Ts.device != T.device: self.test_Ts = self.test_Ts.to(T.device)

        self.test_Xs = torch.cat([self.test_Xs, X])
        self.test_Ts = torch.cat([self.test_Ts, T])

        
        # Y = torch.from_numpy(Y)
        # result = pl.EvalResult()

    def test_epoch_end(self, *args, **kwargs):
        recall = []
        logs = {}
        Y = assign_by_euclidian_at_k(self.test_Xs.cpu(), self.test_Ts.cpu(), 8) # min(8, len(batch)))
        Y = torch.from_numpy(Y)
        for k in [1, 2, 4, 8]:
            r_at_k = 100*calc_recall_at_k(self.test_Ts.cpu(), Y, k)
            recall.append(r_at_k)
            logs[f"R@{k}"] = r_at_k  # f"{r_at_k:.3f}"
            # logging.info("R@{} : {:.3f}".format(k, 100 * r_at_k))
        # result.log_dict(logs)
        # print(logs)
        self.test_Ts = torch.empty(0, dtype=torch.int64)
        self.test_Xs = torch.empty(0, dtype=torch.int64)
        return logs


    # def validation_step(self, batch, batch_idx):
    #     images, target = batch
    #     output = self(images)
    #     loss_val = F.cross_entropy(output, target)
    #     acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))

    #     output = OrderedDict({
    #         'val_loss': loss_val,
    #         'val_acc1': acc1,
    #         'val_acc5': acc5,
    #     })
    #     return output

    # def validation_epoch_end(self, outputs):
    #     tqdm_dict = {}
    #     for metric_name in ["val_loss", "val_acc1", "val_acc5"]:
    #         tqdm_dict[metric_name] = torch.stack([output[metric_name] for output in outputs]).mean()

    #     result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': tqdm_dict["val_loss"]}
    #     return result

    # @staticmethod
    # def __accuracy(output, target, topk=(1,)):
    #     """Computes the accuracy over the k top predictions for the specified values of k"""
    #     with torch.no_grad():
    #         maxk = max(topk)
    #         batch_size = target.size(0)

    #         _, pred = output.topk(maxk, 1, True, True)
    #         pred = pred.t()
    #         correct = pred.eq(target.view(1, -1).expand_as(pred))

    #         res = []
    #         for k in topk:
    #             correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
    #             res.append(correct_k.mul_(100.0 / batch_size))
    #         return res

    def configure_optimizers(self):
        
        optimizer = optim.Adam(
            [
                {
                    "params": self.parameters(),
                    "lr": self.hparams.lr_backbone,
                    "eps": 1.0,
                    "weight_decay": self.hparams.weight_decay_backbone,
                },
                # {
                #     "params": self.backbone.fc.parameters(),
                #     "lr": self.hparams.lr_embedding,
                #     "eps": 1.0,
                #     "weight_decay": self.hparams.weight_decay_embedding,
                # },
                # {
                #     "params": self.proxies,
                #     "lr": self.hparams.lr,
                #     "eps": 1.0,
                #     "weight_decay": self.hparams.weight_decay_proxynca,
                # },
            ],
        )

        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.94)
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, gamma=0.94)

        return [optimizer], [scheduler]



if __name__ == '__main__':

    import random
    nb_classes = 100
    sz_batch = 32
    sz_embedding = 64
    X = torch.randn(sz_batch, sz_embedding).cuda()
    breakpoint()
    P = torch.randn(nb_classes, sz_embedding).cuda()
    T = torch.randint(low=0, high=nb_classes, size=[sz_batch]).cuda()
    criterion = ProxyNCA(nb_classes, sz_embedding).cuda()

    print(criterion(X, T.view(sz_batch)))
