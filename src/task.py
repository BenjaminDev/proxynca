from logging import log
from numpy.core.shape_base import stack
import pytorch_lightning as pl
from pytorch_lightning import EvalResult, TrainResult
import torch
from torch import tensor
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


class DML(pl.LightningModule):
    def __init__(self, model, criterion, **kwargs) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion

        self.save_hyperparameters()
        self.test_Xs = torch.empty(0)
        self.test_Ts = torch.empty(0)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, target, _ = batch
        output = self(images)

        loss = self.criterion(output, target)
        result = TrainResult(loss)
        result.log_dict({"train_loss": loss})
        # result.log_dict({"examples": [wandb.Image(Image.open("/mnt/vol_b/cars/car_ims/014839.jpg"), caption="Label")]})
        return result
        # return {"loss": loss, "train_loss":loss.item() }
    def validation_step(self, batch, batch_idx) -> EvalResult:
        X, T, *_ = self.predict_batchwise(batch)
        return X, T
    def validation_epoch_end(self, outputs)-> EvalResult:
        all_Xs = torch.empty(0).to(self.device)
        all_Ts = torch.empty(0).to(self.device)
        for X_T in outputs:
            X, T = X_T
            all_Xs = torch.cat([all_Xs, X])
            all_Ts = torch.cat([all_Ts, T])
        
        recall = []
        logs = {}
        Y = assign_by_euclidian_at_k(all_Xs, all_Ts, 8) # min(8, len(batch)))
        for k in [1, 2, 4, 8]:
            r_at_k = calc_recall_at_k(all_Ts, Y, k)
            recall.append(r_at_k)
            logs[f"R@{k}"] = r_at_k  # f"{r_at_k:.3f}"
            # logging.info("R@{} : {:.3f}".format(k, 100 * r_at_k))
        # result.log_dict(logs)
        # print(logs)
        # result = EvalResult()

        return logs

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
        Y = assign_by_euclidian_at_k(self.test_Xs, self.test_Ts, 8) # min(8, len(batch)))
        for k in [1, 2, 4, 8]:
            r_at_k = calc_recall_at_k(self.test_Ts, Y, k)
            recall.append(r_at_k)
            logs[f"R@{k}"] = r_at_k  # f"{r_at_k:.3f}"
            # logging.info("R@{} : {:.3f}".format(k, 100 * r_at_k))
        # result.log_dict(logs)
        # print(logs)

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

        parameters = list(
            set(self.model.parameters()).difference(
                set(self.model.embedding_layer.parameters())
            )
        )
        optimizer = optim.Adam(
            [
                # {
                #     "params": parameters,
                #     "lr": self.hparams.lr_backbone,
                #     "weight_decay": self.hparams.weight_decay_backbone,
                # },
                {
                    "params": self.model.embedding_layer.parameters(),
                    "lr": self.hparams.lr_embedding,
                    "weight_decay": self.hparams.weight_decay_embedding,
                },
                {
                    "params": self.criterion.parameters(),
                    "lr": self.hparams.lr_proxynca,
                    "weight_decay": self.hparams.weight_decay_proxynca,
                },
            ],
        )
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.94)
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
