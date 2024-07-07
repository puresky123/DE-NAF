import torch,os,imageio,sys
from tqdm.auto import tqdm
import imageio.v2 as iio
from src.utils import *
from src.render import render, run_network

def OctreeRender_trilinear_fast(rays, denaf, chunk=1024, N_samples=-1, ndc_ray=False, is_train=False, device='cuda'):
    projs, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    N_rays_all = rays.shape[0]
    # idx_chunks = torch.split(torch.arange(N_rays_all), chunk)
    # for chunk_idx in idx_chunks:
        # rays_chunk = rays[chunk_idx].to(device)
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        proj_map = denaf(rays_chunk, is_train=is_train, ndc_ray=ndc_ray, N_samples=N_samples)
        projs.append(proj_map)
    
    return torch.cat(projs)

# 验证过程
@torch.no_grad()
def evaluation(test_dataset, denaf, args, savePath=None, N_vis=5, prtx='', N_samples=-1, device='cuda', compute_extra_metrics=True):
    proj_PSNRs, proj_maps = [], []
    ssims, l_alex, l_vgg = [], [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/proj", exist_ok=True)
    os.makedirs(savePath + "/npy", exist_ok=True)
    # 清除进度条的实例
    try:
        tqdm._instances.clear()
    except Exception:
        pass

    # eval the projection
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.rays.shape[0] // N_vis, 1) # 根据 N_vis 决定评估时用到的样本数量
    idxs = list(range(0, test_dataset.rays.shape[0], img_eval_interval))
    for idx, samples in tqdm(enumerate(test_dataset.rays[0::img_eval_interval]), file=sys.stdout):
        W, H = test_dataset.img_wh
        n_rays = test_dataset.n_rays
        near_far = test_dataset.near_far
        rays = samples.view(-1, samples.shape[-1])
        proj_map = []
        # 一张投影的射线总数为 256*256，循环中每次渲染的射线数为 self.n_rays
        for i in range(0, rays.shape[0], n_rays):
            proj_map.append(render(rays[i:i + n_rays], denaf, n_samples=args.n_samples, perturb=True, netchunk=409600, raw_noise_std=0.)["acc"])
        proj_map = torch.cat(proj_map, 0).reshape(H, W).cpu()
        if len(test_dataset.projs):
            gt_proj = test_dataset.projs[idxs[idx]].view(H, W).cpu()

            proj_PSNRs.append(get_psnr_np(proj_map.numpy(), gt_proj.numpy()))
            if compute_extra_metrics:
                ssim = gray_ssim(proj_map, gt_proj, 1)
                l_a = gray_lpips(gt_proj.numpy(), proj_map.numpy(), 'alex', denaf.device)
                l_v = gray_lpips(gt_proj.numpy(), proj_map.numpy(), 'vgg', denaf.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)
        if savePath is not None:
            show_proj = np.concatenate((proj_map, gt_proj), axis=1)
            cv2.imwrite(f'{savePath}/proj/{prtx}{idx:03d}left_pre_right_gt.png', (cast_to_image(show_proj) * 255).astype(np.uint8))
            proj_maps.append((cast_to_image2(proj_map) * 255).astype(np.uint8))
    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(proj_maps), fps=5, quality=10)
    if proj_PSNRs:
        psnr = np.mean(np.asarray(proj_PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            loss_proj = {"proj_psnr_avg": psnr,
                         "proj_ssim_avg": ssim,
                         "proj_l_a_avg": l_a,
                         "proj_l_v_avg": l_v}
        else:
            loss_proj = {"proj_psnr_avg": psnr}
        with open(os.path.join(savePath, "proj/proj_metrics.txt"), "w") as f:
            for key, value in loss_proj.items():
                f.write("%s: %f\n" % (key, value.item()))

    # eval the CT image
    image = test_dataset.image # [128, 128, 128]
    voxels_pts = test_dataset.voxels # [128, 128, 128, 3]
    # 分块计算，降低显存占用
    n_size1 = voxels_pts.shape[0]
    n_size2 = voxels_pts.shape[1]
    n_size3 = voxels_pts.shape[2]
    voxels_pts = voxels_pts.reshape(-1, 3)  # [n*n*n, 3]
    image_pred = []

    image_pred = run_network(voxels_pts, denaf, netchunk=409600)
    image_pred = image_pred.reshape(n_size1, n_size2, n_size3)

    # 计算CT的 psnr 与 ssim
    psnr_3d = get_psnr_3d(image_pred, image)
    ssim_3d = get_ssim_3d(image_pred, image)  # 与原函数区别: 在structural_similarity()中指定了data_range的范围

    # 保存CT图像
    show_slice = 5
    show_step = image.shape[-1] // show_slice
    show_image = image[..., ::show_step]
    # print(f'show_image.shape: {show_image.shape}')
    show_image_pred = image_pred[..., ::show_step]
    # print(f'show_image_pred.shape: {show_image_pred.shape}')

    # 保存CT图像张量信息
    # np.save(f'{savePath}/npy/image_gt.npy', show_image.cpu().detach().numpy())
    # np.save(f'{savePath}/npy/image_pred.npy', show_image_pred.cpu().detach().numpy())
    # np.save(f'{savePath}/npy/image_gt.npy', image.cpu().detach().numpy())
    np.save(f'{savePath}/npy/image_pred.npy', image_pred.cpu().detach().numpy())

    # 将数据集中CT切片与模型输出的CT切片拼接在一起保存在show中
    show = []
    # dim=0: 按行拼接, dim=1: 按列拼接
    for i_show in range(show_slice):
        show.append(torch.concat([show_image[..., i_show], show_image_pred[..., i_show]], dim=0))
    show_density = torch.concat(show, dim=1)
    cv2.imwrite(os.path.join(savePath, "slice_show_row1_gt_row2_pred.png"), (cast_to_image(show_density) * 255).astype(np.uint8))
    loss_img = {"3d_psnr": psnr_3d,
                "3d_ssim": ssim_3d}
    with open(os.path.join(savePath, "ct_metrics.txt"), "w") as f:
        for key, value in loss_img.items():
            f.write("%s: %f\n" % (key, value.item()))

    return proj_PSNRs, psnr_3d, ssim_3d



