import os
import os.path as osp
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from shutil import copyfile
import numpy as np

from .dataset import TIGREDataset as Dataset
from .network import get_network
from .encoder import get_encoder
from .utils import N_to_reso


class Trainer:
    def __init__(self, cfg, device="cuda"):
        # Args
        self.global_step = 0
        self.conf = cfg
        # self.n_fine = cfg["render"]["n_fine"] # 每条射线上进行精采样的采样点数量
        self.epochs = cfg["train"]["epoch"]
        self.i_eval = cfg["log"]["i_eval"] # Epoch for evaluation
        self.i_save = cfg["log"]["i_save"] # Epoch for evaluation
        # The total number of spatial coordinates for each input network
        self.netchunk = cfg["render"]["netchunk"] # 409600
        self.n_rays = cfg["train"]["n_rays"] # 推理时每轮渲染的射线量
  
        # Log direcotry
        self.expdir = osp.join(cfg["exp"]["expdir"], cfg["exp"]["expname"])
        self.ckptdir = osp.join(self.expdir, "ckpt.tar")
        self.ckptdir_backup = osp.join(self.expdir, "ckpt_backup.tar")
        self.evaldir = osp.join(self.expdir, "eval")
        # "exist_ok=True" 若文件夹不存在则创建，若存在不会抛异常，继续让它存在
        os.makedirs(self.evaldir, exist_ok=True)

        # Dataset
        train_dataset = Dataset(cfg["exp"]["datadir"], self.n_rays, "train", device)
        self.eval_dset = Dataset(cfg["exp"]["datadir"], self.n_rays, "val", device) if self.i_eval > 0 else None
        self.train_dloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg["train"]["n_batch"])
        # get the coordinate of the voxels
        self.voxels = self.eval_dset.voxels if self.i_eval > 0 else None

        # init parameters
        aabb = train_dataset.scene_bbox.to(device)  # 场景边界框 such as torch.tensor([[-0.3, -0.3, -0.3], [0.3, 0.3, 0.3]])
        reso_cur = N_to_reso(cfg["train"]["N_voxel_init"], aabb)  # 根据初始体素的量和场景边界来获取场景的分辨率 2097156——>[128,128,128]
        near_far = train_dataset.near_far

        # Network
        network = get_network(cfg["network"]["net_type"])
        # pop this parameter to match the input parameters of network()
        cfg["network"].pop("net_type", None)
        # encoder = get_encoder(**cfg["encoder"])

        self.net = network(aabb, reso_cur, device, epoch=0, density_n_comp=cfg["network"]["n_lamb_sigma"], density_dim=cfg["network"]["data_dim_sigma"], near_far=near_far,
                   shadingMode=cfg["network"]["shadingMode"], alphaMask_thres=cfg["network"]["alpha_mask_thre"], density_shift=cfg["network"]["ensity_shift"],
                   distance_scale=cfg["network"]["distance_scale"], pos_pe=cfg["network"]["pos_pe"], featureC=cfg["network"]["hidden_dim"], step_ratio=0.5,
                   fea2denseAct=cfg["network"]["fea2denseAct"]).to(device)

        grad_vars = self.net.get_optparam_groups(cfg["train"]["lr_init"], cfg["train"]["lr_basis"])

        # 设置优化器
        self.optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
        # self.optimizer = torch.optim.Adam(params=grad_vars, lr=cfg["train"]["lrate"], betas=(0.9, 0.999))
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=cfg["train"]["lrate_gamma"])

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer, step_size=cfg["train"]["lrate_step"], gamma=cfg["train"]["lrate_gamma"])

        # Load checkpoints
        self.epoch_start = 0
        if cfg["train"]["resume"] and osp.exists(self.ckptdir):
            print(f"Load checkpoints from {self.ckptdir}.")
            ckpt = torch.load(self.ckptdir)
            self.epoch_start = ckpt["epoch"] + 1
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.global_step = self.epoch_start * len(self.train_dloader)
            self.net.load_state_dict(ckpt["network"])
            if self.n_fine > 0:
                self.net_fine.load_state_dict(ckpt["network_fine"])

        # Summary writer
        self.writer = SummaryWriter(self.expdir)
        self.writer.add_text("parameters", self.args2string(cfg), global_step=0)

    def args2string(self, hp):
        # Transfer args to string.
        json_hp = json.dumps(hp, indent=2)
        return "".join("\t" + line for line in json_hp.splitlines(True))

    def start(self):
        # start train
        def fmt_loss_str(losses):
            return "".join(", " + k + ": " + f"{losses[k].item():.3g}" for k in losses)

        # set the progress bar
        iter_per_epoch = len(self.train_dloader)
        pbar = tqdm(total= iter_per_epoch * self.epochs, leave=True)
        if self.epoch_start > 0:
            pbar.update(self.epoch_start*iter_per_epoch)

        # Main loop (epoch)
        for idx_epoch in range(self.epoch_start, self.epochs+1):
            # Evaluate
            if (idx_epoch % self.i_eval == 0 or idx_epoch == self.epochs) and self.i_eval > 0:
                # set eval() when model in inference
                self.net.eval()
                with torch.no_grad():
                    loss_test = self.eval_step(global_step=self.global_step, idx_epoch=idx_epoch)
                # finish inference and set train() to let the model back to training
                self.net.train()
                tqdm.write(f"[EVAL] epoch: {idx_epoch}/{self.epochs}{fmt_loss_str(loss_test)}")

            # Train (iteration)
            # n_batch = 1, 一个data包含一张投影图
            for data in self.train_dloader:
                self.global_step += 1
                # set train() when model in train
                self.net.train()
                loss_train = self.train_step(data, global_step=self.global_step, idx_epoch=idx_epoch)
                pbar.set_description(f"epoch={idx_epoch}/{self.epochs}, loss={loss_train:.3g}, lr={self.optimizer.param_groups[0]['lr']:.3g}")
                pbar.update(1)
            
            # Save
            if (idx_epoch % self.i_save == 0 or idx_epoch == self.epochs) and self.i_save > 0 and idx_epoch > 0:
                if osp.exists(self.ckptdir):
                    copyfile(self.ckptdir, self.ckptdir_backup)
                tqdm.write(f"[SAVE] epoch: {idx_epoch}/{self.epochs}, path: {self.ckptdir}")
                torch.save(
                    {
                        "epoch": idx_epoch,
                        "network": self.net.state_dict(),
                        "network_fine": self.net_fine.state_dict() if self.n_fine > 0 else None,
                        "optimizer": self.optimizer.state_dict(),
                    },
                    self.ckptdir,
                )

            # Update lrate
            self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
            self.lr_scheduler.step()

        tqdm.write(f"Training complete! See logs in {self.expdir}")

    def train_step(self, data, global_step, idx_epoch):
        # Training step
        # in each batch should initialize the parameter gradient of the model to 0
        self.optimizer.zero_grad()
        #
        loss = self.compute_loss(data, global_step, idx_epoch)
        # back propagation to compute the gradient
        loss.backward()
        # update parameters
        self.optimizer.step()
        return loss.item()
        
    def compute_loss(self, data, global_step, idx_epoch):
        # Training step
        raise NotImplementedError()

    def eval_step(self, global_step, idx_epoch):
        # Evaluation step
        raise NotImplementedError()
        