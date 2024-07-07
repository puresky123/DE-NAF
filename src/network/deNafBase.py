import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
import time
from src.encoder import get_encoder
from .network import DensityNetwork
from .apparatus import *

# import hash encoding
encoder = get_encoder(encoding='hashgrid', input_di=3, multires=6, num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19) # hashgrid / frequency

# 暂时没用
def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma*dist)
    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)
    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:,-1:]
# 暂时没没用
def raw2outputs(sigma, dist, raw_noise_std = 0):
    # alpha = 1. - torch.exp(-sigma * dist)
    # 是否给神经网络的预测结果添加噪声
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(sigma[..., 0].shape) * raw_noise_std
        noise = noise.to(sigma.device)
    acc = torch.sum((sigma + noise) * dist, dim=-1) # [4096]
    return acc

class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device
        self.aabb=aabb.to(self.device) # 场景边界框
        self.aabbSize = self.aabb[1] - self.aabb[0] # 计算场景边界框的尺寸
        self.invgridSize = 1.0/self.aabbSize * 2 # 计算逆网格尺寸
        self.alpha_volume = alpha_volume.view(1, 1, *alpha_volume.shape[-3:]) # reshape为[1,1,depth,height,width]的alpha_volume
        # 计算网格的尺寸
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1], alpha_volume.shape[-2], alpha_volume.shape[-3]]).to(self.device)

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled) # 将采样的坐标标准化
        # 在alpha_volume上进行网格采样，获取每个采样点处的alpha值,并将结果展平为一维张量
        # xyz_sampled[ray_valid]: [n, 3] —> [1, n, 1, 1, 3]
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1,-1,1,1,3), align_corners=True).view(-1)
        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invgridSize - 1 # 将采样坐标标准化为范围在[-1, 1]之间的值

class TensorBase(torch.nn.Module):
    def __init__(self, aabb, gridSize, device, epoch = 0, density_n_comp = 8, density_dim = 27, tensor_level = 1, store_way = 'Hash_Tensor', shadingMode = 'Hash_Tensor_MLP1',
                 alphaMask = None, near_far=[2.0, 6.0], density_shift = -10, alphaMask_thres=0.001, distance_scale=25,
                 rayMarch_weight_thres=0.0001, pos_pe = 6, featureC=32, step_ratio=2.0, fea2denseAct = 'softplus'):
        super(TensorBase, self).__init__()
        #
        self.bound = 0.3
        self.epoch = epoch # 用于模型保存时记录当前epoch
        self.tensor_level = tensor_level # the level of multi—scale for Tensor
        self.store_way = store_way # 'Tensor'、'Hash'、'PE'、'Hash_Tensor'
        self.is_mlp = 'True'
        self.density_n_comp = density_n_comp # n_lamb_sigma [16,16,16]
        self.density_dim = density_dim # data_dim_density 27
        self.aabb = aabb # 场景边界框
        self.alphaMask = alphaMask # alpha 掩码
        self.device=device
        self.density_shift = density_shift  # 用于softplus
        self.alphaMask_thres = alphaMask_thres # 设定 alpha mask 的阈值
        self.distance_scale = distance_scale # 用于计算缩放采样距离
        self.rayMarch_weight_thres = rayMarch_weight_thres # mask points in ray marching
        self.fea2denseAct = fea2denseAct # softplus or ReLU or sigmoid
        self.near_far = near_far
        self.step_ratio = step_ratio
        self.update_stepSize(gridSize) # gridSize 即 reso_cur
        # 用于SVD初始化
        self.matMode = [[0, 1], [0, 2], [1, 2]] # for plane init
        self.vecMode =  [2, 1, 0] # for line init
        self.comp_w = [1, 1, 1]
        self.init_svd_volume(gridSize[0], device) # in tensorNAF

        # 根据shadingMode来初始化渲染函数 当前为 MLP_NAF
        self.shadingMode, self.pos_pe, self.featureC = shadingMode, pos_pe, featureC
        self.init_render_func(shadingMode, pos_pe, featureC, device)

    # 根据shadingMode来初始化渲染函数
    def init_render_func(self, shadingMode, pos_pe, featureC, device):
        if shadingMode == 'Tensor_MLP': # 从Tensor中提取存储的场景特征(特征经过mlp降维)并用MLP解码
            self.renderModule = Tensor_Only_Render(self.density_dim, pos_pe, num_layers=4, hidden_dim=featureC, skips=[]).to(device)
        elif shadingMode == 'Hash_MLP': # 从hash表中提取存储的场景特征并用MLP解码(naf的渲染方式)
            self.renderModule = Hash_Only_Render(encoder, bound=0.3, num_layers=4, hidden_dim=32, skips=[2]).to(device)
            # self.renderModule = DensityNetwork(encoder).to(device)
        elif shadingMode == 'PE_MLP':
            self.renderModule = PE_Only_Render(encoder, bound=0.3, num_layers=8, hidden_dim=256, skips=[4], last_activation="relu").to(device)
        elif shadingMode == 'Hash_Tensor_MLP': # 从hash表与tensor中分别提取存储的场景特征并用MLP解码，其中tensor特征经过mlp降维
            self.renderModule = Hash_Tensor_Render(encoder, self.density_dim * self.tensor_level, bound=0.3, num_layers=4, hidden_dim=32, skips=[]).to(device)
        else:
            print("Unrecognized shading module")
            exit()
        # print("pos_pe", pos_pe)
        print(self.renderModule)

    # 根据给定的网格大小（gridSize）和包围盒（aabb），计算并更新步长大小（stepSize）和采样点数（nSamples）
    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize

        # single-scale
        self.gridSize = torch.LongTensor(gridSize).to(self.device) # for ckpt param
        # self.units = self.aabbSize / (self.gridSize - 1)  # 计算场景框中每个单元格的大小，由于场景框的大小是固定的，单元格大小与场景分辨率有关
        # # self.units1 = self.aabbSize / (self.gridSize // 2 - 1)
        # # self.units2 = self.aabbSize / (self.gridSize // 4 - 1)
        # multi-scale
        gridsizes = []
        for i in range(self.tensor_level):
            resolution = []
            for element in gridSize:
                newelement = element * 2 ** i  # 每个分辨率等级的分辨率比前一级提高一倍, base resolution = 64
                resolution.append(newelement)
            gridsizes.append(resolution)
        self.gridsizes = torch.LongTensor(gridsizes).to(self.device)
        self.units = []
        for i in range(self.tensor_level):
            units = self.aabbSize / (self.gridsizes[i] - 1)
            self.units.append(units)
        print(f'different level units: {self.units}')

        self.stepSize=torch.mean(self.units[self.tensor_level-1])*self.step_ratio # 选择最高网格分辨率计算网格单元的大小，从而确定步长
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples=int((self.aabbDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)

    # 初始化奇异值分解（SVD）体积
    def init_svd_volume(self, res, device):
        pass
    # 计算密度特征
    def compute_densityfeature_without_mlp(self, xyz_sampled):
        pass
    # 计算密度特征
    def compute_densityfeature_with_mlp(self, xyz_sampled, index):
        pass
    # 标准化坐标
    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invaabbSize - 1
    # 获取优化器参数组
    def get_optparam_groups(self, lr_init_spatial = 0.02, lr_init_network = 0.001):
        pass

    # 获取关键字参数
    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize': self.gridSize.tolist(),
            'density_n_comp': self.density_n_comp,
            'density_dim': self.density_dim,
            'tensor_level': self.tensor_level,

            'density_shift': self.density_shift,
            'alphaMask_thres': self.alphaMask_thres,
            'distance_scale': self.distance_scale,
            'rayMarch_weight_thres': self.rayMarch_weight_thres,
            'fea2denseAct': self.fea2denseAct,

            'near_far': self.near_far,
            'step_ratio': self.step_ratio,

            'shadingMode': self.shadingMode,
            'store_way': self.store_way,
            'pos_pe': self.pos_pe,
            'featureC': self.featureC
        }

    # 在射线上采样，for tigre
    def naf_sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        bound = 0.3
        n_rays = rays_o.shape[0]
        # rays_o, rays_d = rays[..., :3], rays[..., 3:6]
        near, far = self.near_far
        # N_samples = N_samples if N_samples>0 else self.nSamples
        N_samples = 192
        perturb = True
        # z_vals: 记录了每条射线上采样点的采样间隔，可以等间隔采样，也可以根据perturb选择是否给采样间隔添加随机扰动
        # 从[0, 1]之间等间隔取 n_samples 个采样点
        t_vals = torch.linspace(0., 1., steps=N_samples, device=self.device)
        # 即 z_vals = near + t_vals * (far - near) , 将采样点的相对间距从[0, 1]转换到射线的[near, far]上
        z_vals = near * (1. - t_vals) + far * (t_vals) # [N_samples]
        z_vals = z_vals.expand([n_rays, N_samples])  # 复制z_vals以匹配光线维度 [chunk, N_samples]

        # 之前的采样点是等间隔采样的，通过is_train来控制是否给采样点的采样间隔添加随机扰动
        if is_train:
            # 获取每个采样点采样间隔的扰动范围
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1]) # 计算每两个采样点之间的中点距离起始点的距离
            upper = torch.cat([mids, z_vals[..., -1:]], -1) # 计算每个采样点采样间隔的上限
            lower = torch.cat([z_vals[..., :1], mids], -1) # 计算每个采样点采样间隔的下限
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape, device=lower.device) # 通过torch.rand() 从(0,1)正态分布取随机数作为每个采样点的随机扰动大小
            z_vals = lower + (upper - lower) * t_rand # 根据给出的上下限 [lower, upper] 计算采样点经过随机扰动操作后采样间隔

        # Generates the position of each sampling point on the ray
        # rays_o[..., None, :].shape: (n_rays, 1, 3); rays_d[..., None, :].shape: (n_rays, 1, 3); z_vals[..., :, None].shape: (n_rays, n_samples, 1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [n_rays, n_samples, 3]
        # limited the position of each sampling point on the ray to the range of bound
        mask_outbbox = ((self.aabb[0] + 1e-6 > pts) | (pts > self.aabb[1] - 1e-6)).any(dim=-1)
        pts = pts.clamp(-bound, bound)
        return pts, z_vals, ~mask_outbbox

    # 该方法用于缩小场景的包围盒和体素大小
    def shrink(self, new_aabb, voxel_size):
        pass


    # 该方法用于生成密集的 alpha 值（体素的透明度），它基于网格尺寸 gridSize 生成一组均匀采样点，然后计算每个采样点的 alpha 值. 返回生成的 alpha 值和对应的密集采样点坐标
    @torch.no_grad()
    def getDenseAlpha(self,gridSize=None):
        gridSize = self.gridSize if gridSize is None else gridSize
        # 在每个维度上生成网格点坐标，范围为[0, 1]，形状为gridSize
        samples = torch.stack(torch.meshgrid(torch.linspace(0, 1, gridSize[0]),
                                             torch.linspace(0, 1, gridSize[1]),
                                             torch.linspace(0, 1, gridSize[2]),), -1).to(self.device)
        # 根据samples和边界框计算得到密集的xyz坐标
        dense_xyz = self.aabb[0] * (1-samples) + self.aabb[1] * samples
        # 创建与dense_xyz[...,0]形状相同的alpha张量
        alpha = torch.zeros_like(dense_xyz[...,0])
        for i in range(gridSize[0]):
            # 将dense_xyz的第i行重新调整为形状为(-1, 3)的张量，并调用compute_alpha计算alpha值
            alpha[i] = self.compute_alpha(dense_xyz[i].view(-1,3)).view((gridSize[1], gridSize[2]))
        return alpha, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200,200,200)):
        # 使用getDenseAlpha函数获取gridSize下对应的alpha值和xyz坐标
        alpha, dense_xyz = self.getDenseAlpha(gridSize)
        # 转置dense_xyz的维度0和维度2，并保持连续性
        dense_xyz = dense_xyz.transpose(0,2).contiguous()
        # 对alpha进行范围限制，将其限制在0到1之间，并转置维度0和维度2，并保持连续性
        alpha = alpha.clamp(0,1).transpose(0,2).contiguous()[None,None]
        # 计算总体体素数量
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]
        # 设置池化核大小为3，对alpha进行最大池化，保持原尺寸
        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        # 将大于等于alphaMask_thres的alpha值设置为1，小于alphaMask_thres的alpha值设置为0
        alpha[alpha>=self.alphaMask_thres] = 1
        alpha[alpha<self.alphaMask_thres] = 0
        # 使用AlphaGridMask类创建新的alphaMask
        self.alphaMask = AlphaGridMask(self.device, self.aabb, alpha)
        # 提取alpha大于0.5的有效xyz坐标
        valid_xyz = dense_xyz[alpha>0.5]
        # 计算有效xyz坐标的最小值和最大值，得到新的边界框
        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)
        # 将xyz_min和xyz_max堆叠成新的边界框
        new_aabb = torch.stack((xyz_min, xyz_max))
        # 计算alpha值的总和，并打印bbox边界框和alpha值占比
        total = torch.sum(alpha)
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f"%(total/total_voxels*100))
        return new_aabb

    # 对射线进行过滤操作，提高渲染效率
    @torch.no_grad()
    def filtering_rays(self, all_rays, all_rgbs, N_samples=256, chunk=10240*5, bbox_only=False):
        print('========> filtering rays ...')
        tt = time.time()
        N = torch.tensor(all_rays.shape[:-1]).prod() # 计算所有光线的数量N
        mask_filtered = []  # 创建一个空列表，用于存储过滤后的光线掩码
        idx_chunks = torch.split(torch.arange(N), chunk) # 将索引拆分成大小为chunk的块, 如[0,1,2,3,4,5,6], 按两个一块分，得[0,1],[2,3],[4,5],[6]
        for idx_chunk in idx_chunks:
            # 从all_rays中选择对应的块，并将其移动到指定的设备上
            rays_chunk = all_rays[idx_chunk].to(self.device) # 从all_rays中选择对应的块，并将其移动到指定的设备上
            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6] # 提取光线的原点和方向
            if bbox_only: # 如果bbox_only为True，则仅考虑边界框内的光线
                # 计算每个光线与边界框的交点
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d) # 使vec中不存在0值
                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1)#.clamp(min=near, max=far) # 计算交点中的最小值
                t_max = torch.maximum(rate_a, rate_b).amin(-1)#.clamp(min=near, max=far) # 计算交点中的最大值
                mask_inbbox = t_max > t_min # 判断光线是否在边界框内
            else:
                # 如果bbox_only为False，在alphaMask内采样光线并判断是否在边界框内
                xyz_sampled, _, _ = self.naf_sample_ray(rays_o, rays_d, N_samples=N_samples, is_train=False)
                mask_inbbox= (self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)
            mask_filtered.append(mask_inbbox.cpu()) # 将光线的边界框内掩码添加到列表中
        # 将所有的光线掩码连接起来，并调整形状为与all_rgbs相同
        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])
        # 输出信息：光线过滤完成，所花时间和光线掩码比例
        print(f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}')
        return all_rays[mask_filtered], all_rgbs[mask_filtered] # 返回过滤后的光线和对应的颜色值

    # 将密度特征转换为密度值
    def feature2sigma(self, density_features):
        if self.fea2denseAct == "sigmoid":
            return F.sigmoid(density_features)
        elif self.fea2denseAct == "softplus":
            return F.softplus(density_features + self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)

    # 计算点的 alpha 值
    def compute_alpha(self, xyz_locs):
        #  xyz_locs: [gridSize[1] * gridSize[2], 3]
        if self.alphaMask is not None:
            # 如果alphaMask不为空，则调用alphaMask的sample_alpha方法对xyz_locs进行采样获取alpha值
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            # 根据alpha值判断是否大于0，生成alpha掩码
            alpha_mask = alphas > 0
        else:
            # 如果alphaMask为空，则设置alpha_mask为全1掩码
            alpha_mask = torch.ones_like(xyz_locs[:,0], dtype=bool)
        # 创建与xyz_locs形状相同的sigma张量
        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)
        # 如果alpha_mask中有任意一个元素为True
        if alpha_mask.any():
            # 则根据alpha_mask对xyz_locs进行采样，得到xyz_sampled,其中xyz_locs[alpha_mask]过滤掉了alpha_mask==false的值
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            # for denaf
            sigma_feature = self.compute_densityfeature_with_mlp(xyz_sampled)
            valid_sigma = self.renderModule(xyz_sampled, sigma_feature)
            sigma[alpha_mask] = valid_sigma.view(-1)

            # for naf
            # valid_sigma = self.renderModule(xyz_sampled)
            # sigma[alpha_mask] = valid_sigma.view(-1)
        # 根据sigma和长度计算alpha值，将结果重新调整为xyz_locs的形状
        sigma = sigma.view(xyz_locs.shape[:-1])
        return sigma

    # significant function #
    def forward(self, xyz_sampled, store_way='Hash_Tensor', is_mlp='False'):
        # step1: Get features with your own storage (Tensor、Hash Table、Tensor + Hash Table)
        # step2: Decode with MLP to get the final output

        # Scene representation method 1: Hash Table
        if self.store_way=='Hash' or self.store_way=='PE': # original naf
            valid_sigma = self.renderModule(xyz_sampled)  # [4096*192, 1]
        # Scene representation method 2: Tensor
        elif self.store_way=='Tensor':
            # compute density feature with mlp
            if self.is_mlp == 'True':
                # 计算xyz_sampled的密度特征，得到sigma_feature [4096*192, 27]
                sigma_feature = self.compute_densityfeature_with_mlp(xyz_sampled)
                valid_sigma = self.renderModule(xyz_sampled, sigma_feature)  # [4096*192, 1]
            # compute density feature without mlp
            else:
                # 计算xyz_sampled的密度特征，得到sigma_feature [4096*192, 1]
                sigma_feature = self.compute_densityfeature_without_mlp(xyz_sampled)
                valid_sigma = self.feature2sigma(sigma_feature)  # [4096*192, 1]
        # Scene representation method 3: Tensor + Hash Table
        elif self.store_way=='Hash_Tensor':
            # compute density feature with mlp
            if self.is_mlp == 'True':
                # 计算xyz_sampled的密度特征，得到sigma_feature [4096*192, 27]
                sigma_features = []
                for index in range(self.tensor_level):
                    sigma_feature = self.compute_densityfeature_with_mlp(xyz_sampled, index)
                    sigma_features.append(sigma_feature)
                sigma_features = torch.cat(sigma_features, dim=1)
                valid_sigma = self.renderModule(xyz_sampled, sigma_features)  # [4096*192, 1]
            # compute density feature without mlp
            else:
                # 计算xyz_sampled的密度特征，得到sigma_feature [4096*192]
                sigma_feature = self.compute_densityfeature_without_mlp(xyz_sampled)
                valid_sigma1 = self.feature2sigma(sigma_feature)  # [4096*192]
                valid_sigma = self.renderModule(xyz_sampled, valid_sigma1.view(-1, 1))  # [4096*192, 1]

        return valid_sigma

    # get multi-scale Tensor features, 暂时用不到
    # sigma_feature_all = []
    # sigma_feature_all.append(self.compute_densityfeature_with_mlp(xyz_sampled, flag=0))
    # sigma_feature_all.append(self.compute_densityfeature_with_mlp(xyz_sampled, flag=1))
    # sigma_feature_all.append(self.compute_densityfeature_with_mlp(xyz_sampled, flag=2))
    # sigma_feature_all = torch.cat(sigma_feature_all, dim=1)

