import torch
import pickle
import os
import sys
import numpy as np

from torch.utils.data import DataLoader, Dataset

class Traindataloader:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None
    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total: # 一轮结束，在下一轮开始时 shuffle
            self.ids = torch.LongTensor(np.random.permutation(self.total)) # 将0-self.total打乱顺序
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]

class ConeGeometry(object):
    # Cone beam CT geometry. Note that we convert to meter from millimeter.
    def __init__(self, data):
        # VARIABLE                                          DESCRIPTION                    UNITS
        # -------------------------------------------------------------------------------------
        self.DSD = data["DSD"]/1000 # Distance Source Detector      (m)
        self.DSO = data["DSO"]/1000  # Distance Source Origin        (m)
        # Detector parameters
        self.nDetector = np.array(data["nDetector"])  # number of pixels              (px)
        self.dDetector = np.array(data["dDetector"])/1000  # size of each pixel            (m)
        self.sDetector = self.nDetector * self.dDetector  # total size of the detector    (m)
        # Image parameters
        self.nVoxel = np.array(data["nVoxel"])  # number of voxels              (vx)
        self.dVoxel = np.array(data["dVoxel"])/1000  # size of each voxel            (m)
        self.sVoxel = self.nVoxel * self.dVoxel  # total size of the image       (m)
        # Offsets
        self.offOrigin = np.array(data["offOrigin"])/1000  # Offset of image from origin   (m)
        self.offDetector = np.array(data["offDetector"])/1000  # Offset of Detector            (m)
        # Auxiliary
        self.accuracy = data["accuracy"]  # Accuracy of FWD proj          (vx/sample)  # noqa: E501
        # Mode
        self.mode = data["mode"]  # parallel, cone                ...
        self.filter = data["filter"]

class TIGREDataset(Dataset):
    # TIGRE dataset.
    def __init__(self, path, n_rays=1024, type="train",  num_projs=50, is_stack=False, device="cuda"):
        super().__init__()
        with open(path, "rb") as handle:
            data = pickle.load(handle) # type of data is dict
        self.geo = ConeGeometry(data)
        self.type = type
        self.n_rays = n_rays
        self.near, self.far = self.get_near_far(self.geo)
        self.near_far = [self.near, self.far]
        #
        self.scene_bbox = torch.tensor([[-0.3, -0.3, -0.3], [0.3, 0.3, 0.3]])
        self.is_stack = is_stack
        self.img_wh = self.geo.nDetector
        self.num_projs = num_projs # num_proj 表示要随机选择的投影和射线组的数量
    
        if type == "train":
            self.projs = torch.tensor(data["train"]["projections"], dtype=torch.float32, device=device) # [50, 256, 256]
            angles = data["train"]["angles"]
            rays = self.get_rays(angles, self.geo, device) # [50, 256, 256, 6]
            self.n_samples = data["numTrain"]

            # 关于稀疏投影的设置
            print(f'num_projs: {self.num_projs}')
            if self.num_projs < self.n_samples:
                # 生成随机索引
                random_indices = torch.randperm(self.n_samples)[:self.num_projs]
                # 使用随机索引从 self.projs 中选择投影
                self.projs = self.projs[random_indices]
                print(f'self.projs.shape: {self.projs.shape}')
                # 使用相同的随机索引从 rays 中选择射线组
                rays = rays[random_indices]
                print(f'rays.shape: {rays.shape}')

            self.rays = torch.cat([rays, torch.ones_like(rays[...,:1])*self.near, torch.ones_like(rays[...,:1])*self.far], dim=-1)
            print(f'self.rays.shape: {self.rays.shape}')
            # coords.shape: [geo.nDetector, geo.nDetector, 2]
            # coords = torch.stack(torch.meshgrid(torch.linspace(0, self.geo.nDetector[1] - 1, self.geo.nDetector[1]),
            #                                     torch.linspace(0, self.geo.nDetector[0] - 1, self.geo.nDetector[0]),
            #                                     indexing="ij"), -1)
            # self.coords = torch.reshape(coords, [-1, 2]) # [geo.nDetector * geo.nDetector, 2]
            if not self.is_stack: # 50==len(dataset)
                self.rays = self.rays.view(-1, 8)  # 光线拼接 (50*h*w, 8)
                self.projs = self.projs.view(-1, 1)  # RGB图像拼接 (50*h*w, 1)
                self.trainingSampler = Traindataloader(self.rays.shape[0], self.n_rays)
            else:
                self.rays = self.rays.view(self.rays.shape[0], -1, 8)  # 光线堆叠 (50, h*w, 8)
                self.projs = self.projs.view(self.projs.shape[0], self.projs.shape[1], self.projs.shape[2], 1)  # RGB图像堆叠 (50, h, w, 1)
            self.image = torch.tensor(data["image"], dtype=torch.float32, device=device)
            self.voxels = torch.tensor(self.get_voxels(self.geo), dtype=torch.float32, device=device)
        elif type == "val":
            self.projs = torch.tensor(data["val"]["projections"], dtype=torch.float32, device=device)
            angles = data["val"]["angles"]
            rays = self.get_rays(angles, self.geo, device)
            self.rays = torch.cat([rays, torch.ones_like(rays[...,:1])*self.near, torch.ones_like(rays[...,:1])*self.far], dim=-1)
            self.n_samples = data["numVal"]
            self.image = torch.tensor(data["image"], dtype=torch.float32, device=device)
            self.voxels = torch.tensor(self.get_voxels(self.geo), dtype=torch.float32, device=device)
        
    def __len__(self):
        # self.n_samples 表示训练集或验证集的数据数量
        # return self.n_samples
        if self.type == "train":
            return self.num_projs
        elif self.type == "val":
            return self.n_samples

    def __getitem__(self, index):
        if self.type == "train":
            if not self.is_stack:
                ray_idx = self.trainingSampler.nextids()
                rays, projs = self.rays[ray_idx], self.projs[ray_idx]
            else:
                rays, projs = self.rays[index], self.projs[index]
            out = {"projs": projs, "rays": rays}
        elif self.type == "val":
            rays = self.rays[index] # [256, 256, 8]
            projs = self.projs[index] # [256, 256]
            out = {"projs":projs, "rays":rays}
        return out

    def get_voxels(self, geo: ConeGeometry):
        # Get the voxels
        n1, n2, n3 = geo.nVoxel 
        s1, s2, s3 = geo.sVoxel / 2 - geo.dVoxel / 2

        xyz = np.meshgrid(np.linspace(-s1, s1, n1),
                          np.linspace(-s2, s2, n2),
                          np.linspace(-s3, s3, n3), indexing="ij")
        voxel = np.asarray(xyz).transpose([1, 2, 3, 0])
        return voxel
    
    def get_rays(self, angles, geo: ConeGeometry, device):
        # Get rays given one angle and x-ray machine geometry
        W, H = geo.nDetector
        DSD = geo.DSD
        rays = []
        
        for angle in angles:
            pose = torch.Tensor(self.angle2pose(geo.DSO, angle)).to(device)
            rays_o, rays_d = None, None
            if geo.mode == "cone":
                i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device),
                                    torch.linspace(0, H - 1, H, device=device), indexing="ij")  # pytorch"s meshgrid has indexing="ij"
                uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]
                vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]
                dirs = torch.stack([uu / DSD, vv / DSD, torch.ones_like(uu)], -1)
                rays_d = torch.sum(torch.matmul(pose[:3,:3], dirs[..., None]).to(device), -1) # pose[:3, :3] * 
                rays_o = pose[:3, -1].expand(rays_d.shape)
            elif geo.mode == "parallel":
                i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device),
                                        torch.linspace(0, H - 1, H, device=device), indexing="ij")  # pytorch"s meshgrid has indexing="ij"
                uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]
                vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]
                dirs = torch.stack([torch.zeros_like(uu), torch.zeros_like(uu), torch.ones_like(uu)], -1)
                rays_d = torch.sum(torch.matmul(pose[:3,:3], dirs[..., None]).to(device), -1) # pose[:3, :3] * 
                rays_o = torch.sum(torch.matmul(pose[:3,:3], torch.stack([uu,vv,torch.zeros_like(uu)],-1)[..., None]).to(device), -1) + pose[:3, -1].expand(rays_d.shape)
            else:
                raise NotImplementedError("Unknown CT scanner type!")
            rays.append(torch.concat([rays_o, rays_d], dim=-1))

        return torch.stack(rays, dim=0)

    def angle2pose(self, DSO, angle):
        phi1 = -np.pi / 2
        R1 = np.array([[1.0, 0.0, 0.0],
                    [0.0, np.cos(phi1), -np.sin(phi1)],
                    [0.0, np.sin(phi1), np.cos(phi1)]])
        phi2 = np.pi / 2
        R2 = np.array([[np.cos(phi2), -np.sin(phi2), 0.0],
                    [np.sin(phi2), np.cos(phi2), 0.0],
                    [0.0, 0.0, 1.0]])
        R3 = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0]])
        rot = np.dot(np.dot(R3, R2), R1)
        trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0])
        # np.eye()—对角阵
        T = np.eye(4)
        T[:-1, :-1] = rot
        T[:-1, -1] = trans
        return T

    def get_near_far(self, geo: ConeGeometry, tolerance=0.005):
        # Compute the near and far threshold.
        dist1 = np.linalg.norm([geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2])
        dist2 = np.linalg.norm([geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2])
        dist3 = np.linalg.norm([geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2])
        dist4 = np.linalg.norm([geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2])
        dist_max = np.max([dist1, dist2, dist3, dist4])
        near = np.max([0, geo.DSO - dist_max - tolerance])
        far = np.min([geo.DSO * 2, geo.DSO + dist_max + tolerance])
        return near, far
