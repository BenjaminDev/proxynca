import pytorch_lightning as pl
from pytorch_lightning import EvalResult, TrainResult
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch import optim
from torch.optim import lr_scheduler

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
        self, nb_classes, sz_embedding, smoothing_const=0.1, scaling_x=1, scaling_p=3
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
        # note that compared to proxy nca, positive included in denominator
        loss = torch.sum(-T * F.log_softmax(-D, -1), -1)
        return loss.mean()


class DML(pl.LightningModule):
    def __init__(self, model, criterion, **kwargs) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, target, _ = batch
        output = self(images)
        loss_val = self.criterion(output, target)

        return {"loss": loss_val}

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

        Y = assign_by_euclidian_at_k(X, T, min(8, len(batch)))
        Y = torch.from_numpy(Y)
        # result = pl.EvalResult()
        recall = []
        logs = {}
        for k in [1, 2, 4, 8]:
            r_at_k = calc_recall_at_k(T, Y, k)
            recall.append(r_at_k)
            logs[f"R@{k}"] = r_at_k  # f"{r_at_k:.3f}"
            # logging.info("R@{} : {:.3f}".format(k, 100 * r_at_k))
        # result.log_dict(logs)
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
        optimizer = optim.SGD(
            [
                {
                    "params": parameters,
                    "lr": self.hparams.lr_backbone,
                    "weight_decay": self.hparams.weight_decay_backbone,
                },
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
        # scheduler = config['lr_scheduler']['type'](
        #     opt, **config['lr_scheduler']['args']
        # )
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 30))
        return [optimizer], [scheduler]

    # def train_dataloader(self):
    #     normalize = transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225],
    #     )

    #     train_dir = os.path.join(self.data_path, 'train')
    #     train_dataset = datasets.ImageFolder(
    #         train_dir,
    #         transforms.Compose([
    #             transforms.RandomResizedCrop(224),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             normalize,
    #         ]))

    #     train_loader = torch.utils.data.DataLoader(
    #         dataset=train_dataset,
    #         batch_size=self.batch_size,
    #         shuffle=True,
    #         num_workers=self.workers,
    #     )
    #     return train_loader

    # def val_dataloader(self):
    #     normalize = transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225],
    #     )
    #     val_dir = os.path.join(self.data_path, 'val')
    #     val_loader = torch.utils.data.DataLoader(
    #         datasets.ImageFolder(val_dir, transforms.Compose([
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             normalize,
    #         ])),
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=self.workers,
    #     )
    #     return val_loader

    # def test_dataloader(self):
    #     return self.val_dataloader()

    # def test_step(self, *args, **kwargs):
    #     return self.validation_step(*args, **kwargs)

    # def test_epoch_end(self, *args, **kwargs):
    #     outputs = self.validation_epoch_end(*args, **kwargs)

    #     def substitute_val_keys(out):
    #         return {k.replace('val', 'test'): v for k, v in out.items()}

    #     outputs = {
    #         'test_loss': outputs['val_loss'],
    #         'progress_bar': substitute_val_keys(outputs['progress_bar']),
    #         'log': substitute_val_keys(outputs['log']),
    #     }
    #     return outputs


# if __name__ == '__main__':

#     import random
#     nb_classes = 100
#     sz_batch = 32
#     sz_embedding = 64
#     X = torch.randn(sz_batch, sz_embedding).cuda()
#     P = torch.randn(nb_classes, sz_embedding).cuda()
#     T = torch.randint(low=0, high=nb_classes, size=[sz_batch]).cuda()
#     criterion = ProxyNCA(nb_classes, sz_embedding).cuda()

#     print(pnca(X, T.view(sz_batch)))
