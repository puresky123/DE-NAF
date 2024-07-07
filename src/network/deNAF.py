from .deNafBase import *

class deNafVMSplit(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(deNafVMSplit, self).__init__(aabb, gridSize, device, **kargs)

    # 初始化奇异值分解（SVD）体积
    def init_svd_volume(self, res, device):

        # single-scale
        # # 初始化self.density_plane, self.density_line
        # self.density_plane, self.density_line = self.init_one_svd(self.density_n_comp, self.gridSize, 0.1, device)
        # # 初始化bais矩阵, 输入特征映射到输出特征空间得到 self.basis_mat
        # self.basis_mat = torch.nn.Linear(sum(self.density_n_comp), self.density_dim, bias=False).to(device)

        # multi-scale 用torch.nn.ModuleList()代替list获取对象，可以保存在state_dict里面
        self.density_planes, self.density_lines, self.basis_mats = torch.nn.ModuleList().to(device), torch.nn.ModuleList().to(device), torch.nn.ModuleList().to(device)
        for i in range(self.tensor_level):
            # 初始化self.density_plane, self.density_line
            density_plane, density_line = self.init_one_svd(self.density_n_comp, self.gridsizes[i], 0.1, device)
            self.density_planes.append(density_plane)
            self.density_lines.append(density_line)
            # 初始化bais矩阵, 输入特征映射到输出特征空间得到 self.basis_mat
            basis_mat = torch.nn.Linear(sum(self.density_n_comp), self.density_dim, bias=False).to(device)
            self.basis_mats.append(basis_mat)

    # 初始化张量分解中的线和面
    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            """
            torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))生成一个形状为(1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0])的张量，
            其中的值是从均值为0、标准差为1的正态分布中随机采样得到的;   scale * torch.randn(...)对生成的随机张量进行缩放，将其值乘以scale;
            torch.nn.Parameter(...)将缩放后的张量转换为可训练的模型参数，并将其添加到plane_coef列表中
            """
            plane_coef.append(torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))
            line_coef.append(torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))
        # 通过将 plane_coef 和 line_coef 传递给 torch.nn.ParameterList，可以将列表中的张量转换为模型参数对象，并将其添加到参数列表中
        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)

    # 获取优化器参数组
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        # # single-scale
        # grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz},
        #              {'params': self.density_plane, 'lr': lr_init_spatialxyz},
        #              {'params': self.basis_mat.parameters(), 'lr': lr_init_network}]
        # if isinstance(self.renderModule, torch.nn.Module): # 用于判断当前选择的renderModule是否是torch.nn.Module类的，如果是，则其参数是可学习的，需要加入到grad_vars中
        #     grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]

        # multi-scale
        grad_vars = []
        for i in range(self.tensor_level):
            grad_vars.append({'params': self.density_lines[i], 'lr': lr_init_spatialxyz})
            grad_vars.append({'params': self.density_planes[i], 'lr': lr_init_spatialxyz})
            grad_vars.append({'params': self.basis_mats[i].parameters(), 'lr': lr_init_network})
        if isinstance(self.renderModule, torch.nn.Module):  # 用于判断当前选择的renderModule是否是torch.nn.Module类的，如果是，则其参数是可学习的，需要加入到grad_vars中
            grad_vars += [{'params': self.renderModule.parameters(), 'lr': lr_init_network}]

        return grad_vars

    # 计算给定向量组件之间的差异度
    def vectorDiffs(self, vector_comps):
        # total 用于累积所有向量组件之间的差异度
        total = 0
        for idx in range(len(vector_comps)):
            # 获取当前向量组件的维度信息，n_comp 表示组件的数量，n_size 表示每个组件的大小
            n_comp, n_size = vector_comps[idx].shape[1:-1]
            # 计算当前向量组件之间的点积
            dotp = torch.matmul(vector_comps[idx].view(n_comp,n_size), vector_comps[idx].view(n_comp,n_size).transpose(-1,-2))
            # 从点积矩阵中提取非对角线元素。通过将点积矩阵展平为一维张量，并去除第一个元素（对角线上的元素），
            # 然后重新调整为 (n_comp - 1, n_comp + 1) 的形状，最后去除最后一列以得到非对角线元素。
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
            # 计算非对角线元素的绝对值的平均值
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    # 计算密度和外观向量组件的差异度之和
    def vector_comp_diffs(self):
        return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line)

    # 计算密度组件（线，面）的L1范数之和
    def density_L1(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + torch.mean(torch.abs(self.density_plane[idx])) + torch.mean(torch.abs(self.density_line[idx]))
        return total

    # 计算密度组件（面）的TV_loss之和
    def TV_loss_density(self, reg): # "reg" from "TVLoss" in utils.py
        total = 0

        # # single-scale
        # for idx in range(len(self.density_plane)):
        #     total = total + reg(self.density_plane[idx]) * 1e-2 #+ reg(self.density_line[idx]) * 1e-3

        # multi-scale
        for i in range(self.tensor_level):
            for idx in range(len(self.density_planes[i])):
                total = total + reg(self.density_planes[i][idx]) * 1e-2

        return total

    # 计算外观组件（面）的TV_loss之和
    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) * 1e-2 #+ reg(self.app_line[idx]) * 1e-3
        return total

    # 计算密度特征：无需mlp解码
    def compute_densityfeature_without_mlp(self, xyz_sampled):
        # plane + line basis
        # 从 xyz_sampled 中选择与密度平面相关的坐标，并将它们组合成 coordinate_plane
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        # 从 xyz_sampled 中选择与密度线相关的坐标，并将它们组合成 coordinate_line
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        # 使用一个由零和原始 coordinate_line 张量沿着最后一个维度堆叠而成的张量来修改坐标线，这样做是为了确保该直线经过原点
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        for idx_plane in range(len(self.density_plane)):
            # 使用双线性插值，根据 density_plane 中的平面系数和 coordinate_plane 中的坐标，计算平面系数在指定坐标上的采样值
            plane_coef_point = F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]], align_corners=True).view(-1, *xyz_sampled.shape[:1])
            # 使用线性插值，根据 density_line 中的线系数和 coordinate_line 中的坐标，计算线系数在指定坐标上的采样值
            line_coef_point = F.grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]], align_corners=True).view(-1, *xyz_sampled.shape[:1])
            # 将平面系数和线系数的乘积通过对第0维求和并累积到 sigma_feature 中
            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0) # [n_rays * n_samples, 1]
        return sigma_feature

    # 计算密度特征：需要mlp解码
    def compute_densityfeature_with_mlp(self, xyz_sampled, index = 0):
        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2) # [3, n_rays * n_samples, 1, 2]
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]])) # [3, n_rays * n_samples]
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2) # [3, n_rays * n_samples, 1, 2]
        plane_coef_point, line_coef_point = [], []

        # # single-scale
        # for idx_plane in range(len(self.density_plane)):
        #     plane_coef_point.append(F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]], align_corners=True).view(-1, *xyz_sampled.shape[:1]))  # append([16, n_rays * n_samples])
        #     line_coef_point.append(F.grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]], align_corners=True).view(-1, *xyz_sampled.shape[:1]))  # append([16, n_rays * n_samples])
        # plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)  # [48, n_rays * n_samples], [48, n_rays * n_samples]
        # return self.basis_mat((plane_coef_point * line_coef_point).T)  # [n_rays * n_samples, 48] ——> [n_rays * n_samples, 48] ——> [n_rays * n_samples, 27]

        # multi-scale
        for idx_plane in range(len(self.density_planes[index])):
            plane_coef_point.append(F.grid_sample(self.density_planes[index][idx_plane], coordinate_plane[[idx_plane]], align_corners=True).view(-1, *xyz_sampled.shape[:1])) # append([16, n_rays * n_samples])
            line_coef_point.append(F.grid_sample(self.density_lines[index][idx_plane], coordinate_line[[idx_plane]], align_corners=True).view(-1, *xyz_sampled.shape[:1])) # append([16, n_rays * n_samples])
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point) # [48, n_rays * n_samples], [48, n_rays * n_samples]
        return self.basis_mats[index]((plane_coef_point * line_coef_point).T)

    # 对plane_coef 和 line_coef 进行上采样
    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            # 对平面系数进行上采样。使用双线性插值将 plane_coef[i] 的数据进行上采样，目标尺寸为 (res_target[mat_id_1], res_target[mat_id_0])。
            # 将上采样结果作为新的参数，并将其形式转换为 torch.nn.Paramete
            plane_coef[i] = torch.nn.Parameter(F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear', align_corners=True))
            line_coef[i] = torch.nn.Parameter(F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))

        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):

        # # single-scale
        # self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)
        # self.update_stepSize(res_target)
        # print(f'upsamping to {res_target}')

        # multi-scale
        res_targets = []
        for i in range(self.tensor_level):
            resolution = []
            for element in res_target:
                newelement = element * 2 ** i  # 每个分辨率等级的分辨率比前一级提高一倍, base resolution = 16
                resolution.append(newelement)
            self.density_planes[i], self.density_lines[i] = self.up_sampling_VM(self.density_planes[i], self.density_lines[i], resolution)
        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb # 将新边界框的最小值和最大值分别赋值给 xyz_min 和 xyz_max

        # # single-scale
        # # 计算缩放因子 t_l 和 b_r，它们将用于确定要保留的边界框区域
        # t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        # # 将缩放因子进行四舍五入并转换为整数类型
        # t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        # # 将 b_r 与 self.gridSize 进行逐元素比较，并选择较小的值，以确保不超出网格的边界
        # b_r = torch.stack([b_r, self.gridSize]).amin(0)
        # for i in range(len(self.vecMode)):
        #     mode0 = self.vecMode[i]
        #     # 根据缩小后的边界框区域，更新density_line的数据，只保留缩小后的区域
        #     self.density_line[i] = torch.nn.Parameter(self.density_line[i].data[...,t_l[mode0]:b_r[mode0],:])
        #     # 根据缩小后的边界框区域，更新app_line的数据，只保留缩小后的区域
        #     mode0, mode1 = self.matMode[i]
        #     # 根据缩小后的边界框区域，更新density_plane的数据，只保留缩小后的区域
        #     self.density_plane[i] = torch.nn.Parameter(self.density_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]])

        # multi-scale
        t_ls, b_rs = [], []
        for i in range(self.tensor_level):
            # 计算缩放因子 t_l 和 b_r，它们将用于确定要保留的边界框区域
            t_l, b_r = (xyz_min - self.aabb[0]) / self.units[i], (xyz_max - self.aabb[0]) / self.units[i]
            # 将缩放因子进行四舍五入并转换为整数类型
            if i == 0:
                t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
            else:
                t_l, b_r = torch.floor(torch.floor(t_l)).long(), torch.ceil(b_r).long() + 1
            # 将 b_r 与 self.gridSize 进行逐元素比较，并选择较小的值，以确保不超出网格的边界
            b_r = torch.stack([b_r, self.gridsizes[i]]).amin(0)
            t_ls.append(t_l)
            b_rs.append(b_r)
            for j in range(len(self.vecMode)):
                mode0 = self.vecMode[j]
                # 根据缩小后的边界框区域，更新density_line的数据，只保留缩小后的区域
                self.density_lines[i][j] = torch.nn.Parameter(self.density_lines[i][j].data[..., t_l[mode0]:b_r[mode0], :])
                mode0, mode1 = self.matMode[j]
                # 根据缩小后的边界框区域，更新density_plane的数据，只保留缩小后的区域
                self.density_planes[i][j] = torch.nn.Parameter(self.density_planes[i][j].data[..., t_l[mode1]:b_r[mode1], t_l[mode0]:b_r[mode0]])

        # 检查当前的网格大小与 alphaMask 的网格大小是否相同。如果不相同，则需要校正边界框
        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            # 计算校正因子
            t_l_r, b_r_r = t_ls[0] / (self.gridSize-1), (b_rs[0]-1) / (self.gridSize-1)
            correct_aabb = torch.zeros_like(new_aabb) # 创建一个与新边界框相同形状的零张量
            # 根据校正因子调整边界框的最小值
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            # 根据校正因子调整边界框的最大值
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb) # 打印原始边界框和校正后的边界框
            new_aabb = correct_aabb # 更新边界框为校正后的边界框

        newSize = b_rs[0] - t_ls[0] # 计算新的尺寸大小
        self.aabb = new_aabb # 更新边界框为校正后的边界框
        self.update_stepSize((newSize[0], newSize[1], newSize[2])) # 根据新的尺寸大小更新步长大小

class deNafCP(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(deNafCP, self).__init__(aabb, gridSize, device, **kargs)

    def init_svd_volume(self, res, device):
        # single-scale
        self.density_line = self.init_one_svd(self.density_n_comp[0], self.gridSize, 0.2, device)
        self.basis_mat = torch.nn.Linear(self.density_n_comp[0], self.density_dim, bias=False).to(device)

        # multi-scale 用torch.nn.ModuleList()代替list获取对象，可以保存在state_dict里面
        # self.density_lines, self.basis_mats = torch.nn.ModuleList().to(device), torch.nn.ModuleList().to(device)
        # for i in range(self.tensor_level):
        #     # 初始化self.density_line
        #     density_line = self.init_one_svd(self.density_n_comp[0], self.gridSize, 0.2, device)
        #     self.density_lines.append(density_line)
        #     # 初始化bais矩阵, 输入特征映射到输出特征空间得到 self.basis_mat
        #     basis_mat = torch.nn.Linear(self.density_n_comp[0], self.density_dim, bias=False).to(device)
        #     self.basis_mats.append(basis_mat)

    def init_one_svd(self, n_component, gridSize, scale, device):
        line_coef = []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            line_coef.append(torch.nn.Parameter(scale * torch.randn((1, n_component, gridSize[vec_id], 1))))
        return torch.nn.ParameterList(line_coef).to(device)

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        # single-scale
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz},
                     {'params': self.basis_mat.parameters(), 'lr': lr_init_network}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params': self.renderModule.parameters(), 'lr': lr_init_network}]

        # multi-scale
        # grad_vars = []
        # for i in range(self.tensor_level):
        #     grad_vars.append({'params': self.density_lines[i], 'lr': lr_init_spatialxyz})
        #     grad_vars.append({'params': self.basis_mats[i].parameters(), 'lr': lr_init_network})
        # if isinstance(self.renderModule, torch.nn.Module):  # 用于判断当前选择的renderModule是否是torch.nn.Module类的，如果是，则其参数是可学习的，需要加入到grad_vars中
        #     grad_vars += [{'params': self.renderModule.parameters(), 'lr': lr_init_network}]

        return grad_vars

    # def compute_densityfeature(self, xyz_sampled):
    #     coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
    #     coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
    #     line_coef_point = F.grid_sample(self.density_line[0], coordinate_line[[0]], align_corners=True).view(-1, *xyz_sampled.shape[:1])
    #     line_coef_point = line_coef_point * F.grid_sample(self.density_line[1], coordinate_line[[1]], align_corners=True).view(-1, *xyz_sampled.shape[:1])
    #     line_coef_point = line_coef_point * F.grid_sample(self.density_line[2], coordinate_line[[2]], align_corners=True).view(-1, *xyz_sampled.shape[:1])
    #     sigma_feature = torch.sum(line_coef_point, dim=0)
    #     return sigma_feature

    def compute_densityfeature_with_mlp(self, xyz_sampled, index=0):
        coordinate_line = torch.stack(
            (xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1,
                                                                                                                  1, 2)

        # single-scale
        line_coef_point = F.grid_sample(self.density_line[0], coordinate_line[[0]], align_corners=True).view(-1,
                                                                                                             *xyz_sampled.shape[
                                                                                                              :1])
        line_coef_point = line_coef_point * F.grid_sample(self.density_line[1], coordinate_line[[1]],
                                                          align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.density_line[2], coordinate_line[[2]],
                                                          align_corners=True).view(-1, *xyz_sampled.shape[:1])
        return self.basis_mat(line_coef_point.T)

        # multi-scale
        # line_coef_point = F.grid_sample(self.density_line[index][0], coordinate_line[[0]], align_corners=True).view(-1, *xyz_sampled.shape[:1])
        # line_coef_point = line_coef_point * F.grid_sample(self.density_line[index][1], coordinate_line[[1]], align_corners=True).view(-1, *xyz_sampled.shape[:1])
        # line_coef_point = line_coef_point * F.grid_sample(self.density_line[index][2], coordinate_line[[2]], align_corners=True).view(-1, *xyz_sampled.shape[:1])
        # return self.basis_mat[index](line_coef_point.T)

    @torch.no_grad()
    def up_sampling_Vector(self, density_line_coef, res_target):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            density_line_coef[i] = torch.nn.Parameter(
                F.interpolate(density_line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear',
                              align_corners=True))
        return density_line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        # single-scale
        self.density_line = self.up_sampling_Vector(self.density_line, res_target)
        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

        # multi-scale
        # res_targets = []
        # for i in range(self.tensor_level):
        #     resolution = []
        #     for element in res_target:
        #         newelement = element * 2 ** i  # 每个分辨率等级的分辨率比前一级提高一倍, base resolution = 16
        #         resolution.append(newelement)
        #     self.density_lines[i] = self.up_sampling_VM(self.density_lines[i], resolution)
        # self.update_stepSize(res_target)
        # print(f'upsamping to {res_target}')

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb

        # single-scale
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)
        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(self.density_line[i].data[..., t_l[mode0]:b_r[mode0], :])

        # multi-scale
        # t_ls, b_rs = [], []
        # for i in range(self.tensor_level):
        #     # 计算缩放因子 t_l 和 b_r，它们将用于确定要保留的边界框区域
        #     t_l, b_r = (xyz_min - self.aabb[0]) / self.units[i], (xyz_max - self.aabb[0]) / self.units[i]
        #     # 将缩放因子进行四舍五入并转换为整数类型
        #     if i == 0:
        #         t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        #     else:
        #         t_l, b_r = torch.floor(torch.floor(t_l)).long(), torch.ceil(b_r).long() + 1
        #     # 将 b_r 与 self.gridSize 进行逐元素比较，并选择较小的值，以确保不超出网格的边界
        #     b_r = torch.stack([b_r, self.gridsizes[i]]).amin(0)
        #     t_ls.append(t_l)
        #     b_rs.append(b_r)
        #     for j in range(len(self.vecMode)):
        #         mode0 = self.vecMode[j]
        #         # 根据缩小后的边界框区域，更新density_line的数据，只保留缩小后的区域
        #         self.density_lines[i][j] = torch.nn.Parameter(self.density_lines[i][j].data[..., t_l[mode0]:b_r[mode0], :])

        # 检查当前的网格大小与 alphaMask 的网格大小是否相同。如果不相同，则需要校正边界框
        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize - 1), (b_r - 1) / (self.gridSize - 1)
            # t_l_r, b_r_r = t_ls[0] / (self.gridSize - 1), (b_rs[0] - 1) / (self.gridSize - 1) # multi-scale
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1 - t_l_r) * self.aabb[0] + t_l_r * self.aabb[1]
            correct_aabb[1] = (1 - b_r_r) * self.aabb[0] + b_r_r * self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        # newSize = b_rs[0] - t_ls[0] # multi-scale
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))

    def density_L1(self):
        total = 0
        for idx in range(len(self.density_line)):
            total = total + torch.mean(torch.abs(self.density_line[idx]))
        return total

    def TV_loss_density(self, reg):
        total = 0
        # single-scale
        for idx in range(len(self.density_line)):
            total = total + reg(self.density_line[idx]) * 1e-3

        # multi-scale
        # for i in range(self.tensor_level):
        #     for idx in range(len(self.density_planes[i])):
        #         total = total + reg(self.density_planes[i][idx]) * 1e-3

        return total

    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_line)):
            total = total + reg(self.app_line[idx]) * 1e-3
        return total

