import torch
import torch.nn as nn


def render(rays, net, n_samples, perturb, netchunk, raw_noise_std):
    n_rays = rays.shape[0]
    rays_o, rays_d, near, far = rays[...,:3], rays[...,3:6],  rays[...,6:7], rays[...,7:]
    # z_vals: 记录了每条射线上采样点的采样间隔，可以等间隔采样，也可以根据perturb选择是否给采样间隔添加随机扰动
    # 从[0, 1]之间等间隔取 n_samples 个采样点
    t_vals = torch.linspace(0., 1., steps=n_samples, device=near.device)
    # 即 z_vals = near + t_vals * (far - near) , 将采样点的相对间距从[0, 1]转换到射线的[near, far]上
    # z_vals.shape: (256*256, 192)
    z_vals = near * (1. - t_vals) + far * (t_vals)
    # 经过代码验证，这一句可以注释掉，前面的计算通过广播已经对张量z_val维度的尺寸进行了的拓展
    z_vals = z_vals.expand([n_rays, n_samples])

    # 之前的采样点是等间隔采样的，通过perturb来控制是否给采样点的采样间隔添加随机扰动
    if perturb:
        # 获取每个采样点采样间隔的扰动范围
        # 计算每两个采样点之间的中点距离起始点的距离
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        # 计算每个采样点采样间隔的上限
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        # 计算每个采样点采样间隔的下限
        lower = torch.cat([z_vals[..., :1], mids], -1)

        # stratified samples in those intervals
        # 通过torch.rand() 从(0,1)正态分布取随机数作为每个采样点的随机扰动大小
        t_rand = torch.rand(z_vals.shape, device=lower.device)
        # 根据给出的上下限 [lower, upper] 计算采样点经过随机扰动操作后采样间隔
        z_vals = lower + (upper - lower) * t_rand

    # Generates the position of each sampling point on the ray
    # rays_o[..., None, :].shape: (n_rays, 1, 3); rays_d[..., None, :].shape: (n_rays, 1, 3); z_vals[..., :, None].shape: (n_rays, n_samples, 1)
    # pts.shape: (n_rays, n_samples, 3)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [n_rays, n_samples, 3]
    # 设置渲染场景的范围框的边界
    bound = net.bound - 1e-6
    # limited the position of each sampling point on the ray to the range of bound
    pts = pts.clamp(-bound, bound)
    # Each point on the ray is put into MLP network_fn and propagated forward to obtain the corresponding (μ) of each point
    raw = run_network(pts, net, netchunk)
    acc, weights = raw2outputs(raw, z_vals, rays_d, raw_noise_std)

    ret = {"acc": acc, "pts":pts}

    # 检查ret中是否有空值或无限大的值
    for k in ret:
        if torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any():
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret

def run_network(inputs, fn, netchunk):
    # Prepares inputs and applies network "fn"
    # uvt_flat.shape: (n_rays * n_samples, 3)
    uvt_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    # netchunk: 每次输入网络的采样点数量; out_flat.shape: (n_rays * n_samples, 1)
    out_flat = torch.cat([fn(uvt_flat[i:i + netchunk]) for i in range(0, uvt_flat.shape[0], netchunk)], 0)
    # out.shape: (n_rays, n_samples, 1)
    out = out_flat.reshape(list(inputs.shape[:-1]) + [out_flat.shape[-1]])
    return out 

# 将模型的输出转换为有意义值
def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0.):
    """
    Transforms model's predictions to semantically meaningful values
    Args:
        raw: [num_rays, num_samples along ray, 1]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    # 计算每条射线上每个采样点之间的间隔, dists.shape: (n_rays, n_samples - 1)
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # (n_rays, n_samples - 1) ——> (n_rays, n_samples)
    dists = torch.cat([dists, torch.Tensor([1e-10]).expand(dists[..., :1].shape).to(dists.device)], -1)
    # 将Z轴之间的距离转换为实际距离; torch.norm(): 求范数
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # 是否给神经网络的预测结果添加噪声
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 0].shape) * raw_noise_std
        noise = noise.to(raw.device)
    # This is different from the formula in the paper because the raw X-ray images were preprocessed before being used
    acc = torch.sum((raw[..., 0] + noise) * dists, dim=-1)

    if raw.shape[-1] == 1:
        # eps.shape: (n_rays, 1, 1)
        eps = torch.ones_like(raw[:, :1, -1]) * 1e-10
        weights = torch.cat([eps, torch.abs(raw[:, 1:, -1] - raw[:, :-1, -1])], dim=-1)
        weights = weights / torch.max(weights)
    elif raw.shape[-1] == 2: # with jac
        weights = raw[..., 1] / torch.max(raw[..., 1])
    else:
        raise NotImplementedError("Wrong raw shape")

    return acc, weights


def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    u = u.contiguous().to(cdf.device)
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


        
        





