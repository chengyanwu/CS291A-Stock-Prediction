
import torch
from torch import nn
import pickle

from .core import Callback

# Cell


class PatchCB(Callback):

    def __init__(self, patch_len, stride):
        """
        Callback used to perform patching on the batch input data
        Args:
            patch_len:        patch length
            stride:           stride
        """
        self.patch_len = patch_len
        self.stride = stride

    def before_forward(self): self.set_patch()

    def set_patch(self):
        """
        take xb from learner and convert to patch: [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
        """
        xb_patch, num_patch = create_patch(
            self.xb, self.patch_len, self.stride)    # xb: [bs x seq_len x n_vars]
        # learner get the transformed input
        # xb_patch: [bs x num_patch x n_vars x patch_len]
        self.learner.xb = xb_patch


class PatchMaskCB(Callback):
    def __init__(self, patch_len, stride, mask_ratio,
                 mask_when_pred: bool = False, augment=False):
        """
        Callback used to perform the pretext task of reconstruct the original data after a binary mask has been applied.
        Args:
            patch_len:        patch length
            stride:           stride
            mask_ratio:       mask ratio
        """
        self.patch_len = patch_len
        self.stride = stride
        self.mask_ratio = mask_ratio
        self.augment = augment

        self.ybs = []
        self.xbs = []

    def before_fit(self):
        # overwrite the predefined loss function
        self.learner.loss_func = self._loss
        device = self.learner.device

    def before_forward(self): self.patch_masking()

    def after_fit(self):
        self.learner.new_data = (self.xbs, self.ybs)

        if self.augment:
            with open('./augment.ob', 'wb') as fp:
                pickle.dump(self.learner.new_data, fp)

    def patch_masking(self):
        """
        xb: [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
        """
        xb_patch, num_patch = create_patch(
            self.xb, self.patch_len, self.stride)    # xb_patch: [bs x num_patch x n_vars x patch_len]
        # xb_mask: [bs x num_patch x n_vars x patch_len]
        xb_mask, _, self.mask, _ = random_masking(xb_patch, self.mask_ratio)
        self.mask = self.mask.bool()    # mask: [bs x num_patch x n_vars]
        self.ybs.append(self.learner.yb)
        self.learner.xb = xb_mask       # learner.xb: masked 4D tensor
        self.learner.yb = xb_patch      # learner.yb: non-masked 4d tensor

    def _loss(self, preds, target):
        """
        preds:   [bs x num_patch x n_vars x patch_len]
        targets: [bs x num_patch x n_vars x patch_len] 
        """
        bs, np, nv, pl = preds.shape
        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * self.mask).sum() / self.mask.sum()
        if self.augment:
            self.xbs.append(preds.permute(0, 1, 3, 2).reshape(bs, np * pl, nv))
        return loss


def create_patch(xb, patch_len, stride):
    """
    xb: [bs x seq_len x n_vars]
    """
    seq_len = xb.shape[1]
    num_patch = (max(seq_len, patch_len)-patch_len) // stride + 1
    tgt_len = patch_len + stride*(num_patch-1)
    s_begin = seq_len - tgt_len

    # xb: [bs x tgt_len x nvars]
    xb = xb[:, s_begin:, :]
    # xb: [bs x num_patch x n_vars x patch_len]
    xb = xb.unfold(dimension=1, size=patch_len, step=stride)
    return xb, num_patch


class Patch(nn.Module):
    def __init__(self, seq_len, patch_len, stride):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patch = (max(seq_len, patch_len)-patch_len) // stride + 1
        tgt_len = patch_len + stride*(self.num_patch-1)
        self.s_begin = seq_len - tgt_len

    def forward(self, x):
        """
        x: [bs x seq_len x n_vars]
        """
        x = x[:, self.s_begin:, :]
        # xb: [bs x num_patch x n_vars x patch_len]
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        return x


def random_masking(xb, mask_ratio):
    # xb: [bs x num_patch x n_vars x patch_len]
    bs, L, nvars, D = xb.shape
    x = xb.clone()

    len_keep = int(L * (1 - mask_ratio))

    # noise in [0, 1], bs x L x nvars
    noise = torch.rand(bs, L, nvars, device=xb.device)

    # sort noise for each sample
    # ascend: small is keep, large is remove
    ids_shuffle = torch.argsort(noise, dim=1)
    # ids_restore: [bs x L x nvars]
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    # ids_keep: [bs x len_keep x nvars]
    ids_keep = ids_shuffle[:, :len_keep, :]
    # x_kept: [bs x len_keep x nvars  x patch_len]
    x_kept = torch.gather(
        x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))

    # removed x
    # x_removed: [bs x (L-len_keep) x nvars x patch_len]
    x_removed = torch.zeros(bs, L-len_keep, nvars, D, device=xb.device)
    # x_: [bs x L x nvars x patch_len]
    x_ = torch.cat([x_kept, x_removed], dim=1)

    # combine the kept part and the removed one
    # x_masked: [bs x num_patch x nvars x patch_len]
    x_masked = torch.gather(
        x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    # mask: [bs x num_patch x nvars]
    mask = torch.ones([bs, L, nvars], device=x.device)
    mask[:, :len_keep, :] = 0
    # unshuffle to get the binary mask
    # [bs x num_patch x nvars]
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, x_kept, mask, ids_restore


def random_masking_3D(xb, mask_ratio):
    # xb: [bs x num_patch x dim]
    bs, L, D = xb.shape
    x = xb.clone()

    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(bs, L, device=xb.device)  # noise in [0, 1], bs x L

    # sort noise for each sample
    # ascend: small is keep, large is remove
    ids_shuffle = torch.argsort(noise, dim=1)
    # ids_restore: [bs x L]
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    # ids_keep: [bs x len_keep]
    ids_keep = ids_shuffle[:, :len_keep]
    # x_kept: [bs x len_keep x dim]
    x_kept = torch.gather(
        x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # removed x
    # x_removed: [bs x (L-len_keep) x dim]
    x_removed = torch.zeros(bs, L-len_keep, D, device=xb.device)
    # x_: [bs x L x dim]
    x_ = torch.cat([x_kept, x_removed], dim=1)

    # combine the kept part and the removed one
    # x_masked: [bs x num_patch x dim]
    x_masked = torch.gather(
        x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    # mask: [bs x num_patch]
    mask = torch.ones([bs, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    # [bs x num_patch]
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, x_kept, mask, ids_restore


if __name__ == "__main__":
    bs, L, nvars, D = 2, 20, 4, 5
    xb = torch.randn(bs, L, nvars, D)
    xb_mask, mask, ids_restore = create_mask(xb, mask_ratio=0.5)
    breakpoint()
