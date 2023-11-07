from utils.core_utils import resample, resample_1d
import torch
import random
import torch.nn.functional as F
from timm.utils import accuracy


class TestEquivarinecError():
    def __init__(self, model) -> None:
        self.model = model

    def get_equivarience_error(self, data_loader, scale_range, samples=100, mode='ideal'):
        device = self.model.device if hasattr(self.model, 'device') else 'cuda'
        if not hasattr(self.model, 'device'):
            self.model = self.model.to('cuda')
        scale_range.sort(reverse=True)
        cur_sam = 0
        total_error = 0.0
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 3:
                    resolution = batch[2][0].item()
                else:
                    resolution = batch[0].shape[-1]
                if resolution != scale_range[0]:
                    continue
                img, y = batch[0], batch[1]
                feature_h = self.model.get_feature(img.to(device))
                t_error = None
                for j in range(1, len(scale_range)):

                    if mode == 'ideal':
                        imgl = resample(
                            img, (scale_range[j], scale_range[j]), complex=False)
                    else:
                        imgl = F.interpolate(img, size=(
                            scale_range[j], scale_range[j]), mode=mode, antialias=True)

                    feature_l = self.model.get_feature(imgl.to(device))

                    if mode == 'ideal':
                        ds_feature_h = resample(
                            feature_h, (feature_l.shape[-2], feature_l.shape[-1]), complex=True, skip_nyq=True)
                        feature_l = resample(
                            feature_l, (feature_l.shape[-2], feature_l.shape[-1]), complex=True, skip_nyq=True)
                    else:
                        ds_feature_h = F.interpolate(torch.real(feature_h), size=(
                            feature_l.shape[-2], feature_l.shape[-1]), mode=mode, antialias=True)
                        feature_l = F.interpolate(torch.real(feature_l), size=(
                            feature_l.shape[-2], feature_l.shape[-1]), mode=mode, antialias=True)

                    error = torch.norm(ds_feature_h - feature_l, dim=(-1, -2), p=2)**2/(
                        torch.norm(ds_feature_h, dim=(-1, -2), p=2)**2+1e-6)

                    error = torch.mean(error, dim=1)  # mean accross channels
                    if t_error is None:
                        t_error = error
                    else:
                        t_error = t_error+error

                total_error += (t_error.sum()/len(scale_range))
                cur_sam += img.shape[0]
                if cur_sam > samples:
                    break

        return total_error/cur_sam


class GetAccuracy():
    def __init__(self, model, data_loader) -> None:
        self.model = model
        self.data_loader = data_loader

    def get_res_acc(self, res):
        device = self.model.device
        self.model.eval()
        acc = {}
        total = {}
        for i in res:
            acc[i] = 0
            total[i] = 0

        with torch.no_grad():
            for i, batch in enumerate(self.data_loader):
                if len(batch) == 3:
                    resolution = batch[2][0].item()
                else:
                    resolution = batch[0].shape[-1]
                if resolution not in res:
                    continue
                batch = (batch[0].to(device), batch[1].to(device))
                logit, loss = self.model._forward_step(
                    batch, i, stage='evaluate', sync_dist=False)
                acc[resolution] += batch[0].shape[0] * \
                    accuracy(logit, batch[1])[0]/100
                total[resolution] += batch[0].shape[0]

        Acc = {}

        for i in res:
            if total[i] == 0:
                total[i] = 1
            Acc[i] = acc[i]/total[i]

        return Acc

    def get_acc(self,):
        device = self.model.device
        self.model.eval()
        acc = 0
        total = 0
        with torch.no_grad():
            for i, batch in enumerate(self.data_loader):
                batch = (batch[0].to(device), batch[1].to(device))
                logit, loss = self.model._forward_step(
                    batch, i, stage='evaluate', sync_dist=False)
                acc += batch[0].shape[0]*accuracy(logit, batch[1])[0]/100
                total += batch[0].shape[0]
        return acc/total
