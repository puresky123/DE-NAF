import os
import torch
from tqdm.auto import tqdm
from opt import config_parser
import json, random
import ast
from eval_render import *
from torch.utils.tensorboard import SummaryWriter
import datetime
import math
import sys
import numpy as np
from src.utils import *
from src.dataset import TIGREDataset as Dataset
from src.network import get_network
from src.network.deNafBase import AlphaGridMask
from src.render import render, run_network
from src.encoder import get_encoder
from src.model_components import S3IM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

@torch.no_grad()
def render_test(args):
    # load dataset
    test_dataset = Dataset(args.datadir, n_rays=args.batch_size, type='val', is_stack=True, device=device)
    ndc_ray = args.ndc_ray
    # check the ckpt exist or not
    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return
    # load ckpt
    network = get_network(args.net_type)
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    denaf = network(**kwargs)
    if 'alphaMask.aabb' in ckpt.keys():
        length = np.prod(ckpt['alphaMask.shape'])
        alpha_volume = torch.from_numpy(np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
        denaf.alphaMask = AlphaGridMask(device, ckpt['alphaMask.aabb'].to(device), alpha_volume.float().to(device))
    denaf.load_state_dict(ckpt['state_dict'])

    # logfolder = os.path.dirname(args.ckpt) # 获取ckpt文件所在的目录，目的是把测试结果和ckpt保存在同一个文件夹
    logfolder = f'{args.evaldir}/{args.expname}_{args.num_projs}_{args.n_lamb_sigma[0]}_{args.epochs}' # 将测试结果保存到专门的路径中

    # 渲染训练集
    if args.render_train:
        os.makedirs(f'{logfolder}', exist_ok=True)
        train_dataset = Dataset(args.datadir, n_rays=1024, type='train', is_stack=True, device=device)
        PSNRs_test, psnr_3d_test, ssim_3d_test = evaluation(train_dataset, denaf, args, f'{logfolder}/imgs_train_all/',
                                                            N_vis=-1, N_samples=-1, device=device, compute_extra_metrics=False)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)}, 3d_psnr: {psnr_3d_test}, 3d_ssim: {ssim_3d_test} <======')

    # 渲染测试集
    if args.render_test:
        os.makedirs(f'{logfolder}', exist_ok=True)
        PSNRs_test, psnr_3d_test, ssim_3d_test = evaluation(test_dataset, denaf, args, f'{logfolder}/imgs_test_all/',
                                                            N_vis=1, N_samples=-1, device=device, compute_extra_metrics=False)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)}, 3d_psnr: {psnr_3d_test}, 3d_ssim: {ssim_3d_test} <======')

def reconstruction(args, np_init_seed, python_init_seed):
    # add S3IM
    s3im_func = S3IM(kernel_size=args.s3im_kernel, stride=args.s3im_stride, repeat_time=args.s3im_repeat_time,
                     patch_height=args.s3im_patch_height, patch_width=args.s3im_patch_width).cuda()

    # load dataset
    train_dataset = Dataset(args.datadir, args.batch_size, "train",  args.num_projs, is_stack=False, device=device)
    # test_dataset = Dataset(args.datadir, args.batch_size, "val", is_stack=True, device=device)
    train_dloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    near_far = train_dataset.near_far # [0.9044903320081219, 1.0955096679918779]

    # init resolution
    upsamp_list = args.upsamp_list # default [2000,3000,4000,5500]

    update_AlphaMask_list = args.update_AlphaMask_list # default [2500]
    #
    n_lamb_sigma = args.n_lamb_sigma # such as [16,16,16]

    # 在保存训练结果时是否记录当前时间
    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}_{args.n_lamb_sigma[0]}_{pow(args.N_voxel_final, 1/3)}_{args.epochs}'
    else:
        logfolder = f'{args.basedir}/{args.expname}_{args.num_projs}_{args.n_lamb_sigma[0]}_{math.ceil(args.N_voxel_final**(1/3))}_{args.epochs}'

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True) # 保存训练中的验证数据
    os.makedirs(f'{logfolder}/ckpt', exist_ok=True) # 保存预训练模型
    summary_writer = SummaryWriter(logfolder)

    # init parameters
    aabb = train_dataset.scene_bbox.to(device) # 场景边界框 such as torch.tensor([[-0.3, -0.3, -0.3], [0.3, 0.3, 0.3]])
    reso_cur = N_to_reso(args.N_voxel_init, aabb)  # 根据初始体素的量和场景边界来获取场景的分辨率 2097156——>[128,128,128]
    # 暂时不用自适应的 nSamples，用固定值 n_samples
    nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio)) # 自适应获取每条射线上的采样点数量，取默认值和自适应值中的最小值
    n_samples = args.n_samples
    global_step = 0 # 即 iteration
    epoch_start = 0 # 训练开始时的 epoch

    # load model
    network = get_network(args.net_type)
    # denaf_network
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        epoch_start = ckpt["epoch"] + 1
        global_step = epoch_start * len(train_dloader)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device})
        denaf = network(**kwargs)

        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            denaf.alphaMask = AlphaGridMask(device, ckpt['alphaMask.aabb'].to(device), alpha_volume.float().to(device))
        denaf.load_state_dict(ckpt['state_dict'])
    else:
        denaf = network(aabb, reso_cur, device, epoch=0, density_n_comp=n_lamb_sigma, density_dim=args.data_dim_sigma, tensor_level=args.tensor_level, near_far=near_far,
                          store_way=args.store_way, shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre, density_shift=args.density_shift, distance_scale=args.distance_scale,
                          pos_pe=args.pos_pe, featureC=args.featureC, step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct)

    # 获取优化器参数组
    grad_vars = denaf.get_optparam_groups(args.lr_init, args.lr_basis)
    print(grad_vars) # 在终端显示获取到的优化器参数组
    
    # 计算模型参数数量 方法1
    total_params = sum(p.numel() for p in denaf.parameters())
    total_params_in_M = total_params / 1_000_000.0
    print(f"Total Parameters in model: {total_params_in_M:.2f} M")

    # 学习率衰减部分
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)
    print(f"lr decay: {args.lr_decay_target_ratio}, {args.lr_decay_iters}, {args.epochs}")

    # 设置优化器
    optimizer = torch.optim.Adam(params=grad_vars, betas=(0.9,0.99))

    # N_voxel_list包含来一系列体素数目的值，这些值是从初始体素数目到最终体素数目之间的指数间隔（linear in logrithmic space）计算得到的，并且去除了初始体素的值
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:]

    # 释放一些持有但未使用的缓存空间
    torch.cuda.empty_cache()
    PSNRs,PSNRs_test = [],[0]

    # if not args.ndc_ray:
    #     allrays, allprojs = denaf.filtering_rays(allrays, allprojs, bbox_only=True)

    # loss weight
    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight: ", Ortho_reg_weight)
    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight: ", L1_reg_weight)
    TV_weight_density = args.TV_weight_density
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density}")
    s3im_weight = args.s3im_weight
    print(f"initial s3im_weight: {s3im_weight}")

    # set the progress bar
    iter_per_epoch = len(train_dloader)
    total_iters = iter_per_epoch * args.epochs
    pbar = tqdm(total=total_iters, leave=True)
    if epoch_start > 0:
        pbar.update(epoch_start * iter_per_epoch)
    # Main loop (epoch)
    for idx_epoch in range(epoch_start, args.epochs):
        for data in train_dloader:
            global_step += 1
            denaf.train() # set train() when model in train

            rays_train = data["rays"].reshape(-1, 8)
            projs_train_gt = data["projs"].reshape(-1, 1)
            projs_map = render(rays_train, denaf, n_samples, perturb=True, netchunk=args.netchunk, raw_noise_std=0.)["acc"]
            projs_map = projs_map.view(-1, 1)  # [batch_size, 1]

            optimizer.zero_grad()
            # 开始计算总损失
            total_loss = 0
            # 计算MSE损失
            loss_mse =  {"loss": 0.}
            calc_mse_loss(loss_mse, projs_map, projs_train_gt)
            #
            total_loss += loss_mse["loss"]
            # 计算向量组件差异度损失
            if Ortho_reg_weight > 0:
                loss_reg = denaf.vector_comp_diffs()
                total_loss += Ortho_reg_weight * loss_reg
                summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=global_step)
            # 计算向量组件L1损失
            if L1_reg_weight > 0:
                loss_reg_L1 = denaf.density_L1()
                total_loss += L1_reg_weight * loss_reg_L1
                summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=global_step)
            # 计算TV_Loss
            if TV_weight_density > 0:
                TV_weight_density *= lr_factor
                loss_tv = denaf.TV_loss_density(tvreg) * TV_weight_density
                total_loss = total_loss + loss_tv
                summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=global_step)
            # 计算S3IM_Loss
            if s3im_weight > 0:
                s3im_pp = s3im_weight * s3im_func(projs_map, projs_train_gt)
                total_loss += s3im_pp

            # optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            psnr = get_psnr(projs_map, projs_train_gt)
            PSNRs.append(psnr.detach().item())
            summary_writer.add_scalar('train/psnr', PSNRs[-1], global_step=global_step)
            summary_writer.add_scalar('train/mse', loss_mse["loss_mse"], global_step=global_step)

            pbar.set_description(
                f'epoch = {idx_epoch}/{args.epochs}:'
                + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                + f' mse = {loss_mse["loss"]:.7f}'
            )
            PSNRs = []
            pbar.update(1)

            # if global_step in update_AlphaMask_list:
            #     if reso_cur[0] * reso_cur[1] * reso_cur[2] < 300**3: # update volume resolution
            #         reso_mask = reso_cur
            #     new_aabb = denaf.updateAlphaMask(tuple(reso_mask))
            #     if global_step == update_AlphaMask_list[0]:
            #         denaf.shrink(new_aabb)
            #         L1_reg_weight = args.L1_weight_rest
            #         print("continuing L1_reg_weight", L1_reg_weight)
            #
            #     if not args.ndc_ray and global_step == update_AlphaMask_list[1]:
            #         # filter rays outside the bbox
            #         allrays, allprojs = denaf.filtering_rays(allrays, allprojs)
            #         trainingSampler = SimpleSampler(allprojs.shape[0], args.batch_size)

            # 当iteration在upsamp_list中
            if global_step in upsamp_list:
                n_voxels = N_voxel_list.pop(0)
                reso_cur = N_to_reso(n_voxels, denaf.aabb)
                nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))
                denaf.upsample_volume_grid(reso_cur)  # 对体素网格进行上采样，即对张量分解组件进行上采样
                if args.lr_upsample_reset:
                    print("reset lr to initial")
                    lr_scale = 1
                else:
                    lr_scale = args.lr_decay_target_ratio ** (global_step / total_iters)
                grad_vars = denaf.get_optparam_groups(args.lr_init * lr_scale, args.lr_basis * lr_scale)
                optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

        # 更新优化器学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # 每个epoch结束在denaf模型中记录当前epoch用于ckpt的保存
        denaf.epoch = idx_epoch

    # 训练结束，保存训练完之后的模型
    path = f'{logfolder}/ckpt/{args.expname}_{args.n_lamb_sigma[0]}_{denaf.gridSize[0]}_{denaf.epoch}.th'
    kwargs = denaf.get_kwargs()
    ckpt = {'epoch': denaf.epoch, 'kwargs': kwargs, 'state_dict': denaf.state_dict()}
    if denaf.alphaMask is not None:
        alpha_volume = denaf.alphaMask.alpha_volume.bool().cpu().numpy()
        ckpt.update({'alphaMask.shape':alpha_volume.shape})
        ckpt.update({'alphaMask.mask':np.packbits(alpha_volume.reshape(-1))})
        ckpt.update({'alphaMask.aabb': denaf.alphaMask.aabb.cpu()})
    torch.save(ckpt, path)

    # 保存随机数种子，用于后续复现训练结果，暂时未完全实现
    seed_state = {
        'python_seed': python_init_seed, # random.getstate()[1][0]
        'torch_seed': torch.initial_seed(),
        'torch_cuda_seed': torch.cuda.initial_seed() if torch.cuda.is_available() else None,
        'numpy_init_seed': np_init_seed,
        'numpy_seed': np.random.get_state()
    }
    # 将随机数种子保存到json文件中
    with open(os.path.join(f'{logfolder}', 'random_seed_state.json'), 'w') as f:
        json.dump(seed_state, f, default=convert_to_json_friendly)

    # 是否渲染测试集
    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        test_dataset = Dataset(args.datadir, args.batch_size, "val", is_stack=True, device=device)
        PSNRs_test, psnr_3d_test, ssim_3d_test = evaluation(test_dataset, denaf, args, f'{logfolder}/imgs_test_all/', N_vis=10, N_samples=-1, device=device, compute_extra_metrics=False)
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=global_step)
        print(f'======> {args.expname} test all psnr_naf: {np.mean(PSNRs_test)}, 3d_psnr: {psnr_3d_test}, 3d_ssim: {ssim_3d_test} <======')

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32) # 设置pytorch中浮点数的默认类型

    args = config_parser() # get arguments
    print(args)

    if os.path.exists('random_seed/random_seed_state.json') and args.get_seed:
        print(f'get store seed')
        # 从json文件中加载保存的种子
        with open('random_seed/random_seed_state.json', 'r') as f:
            seed_state = json.load(f, object_hook=convert_from_json_friendly)
        # 设置随机数种子
        python_init_seed = seed_state['python_seed']
        print(f'python_init_seed: {python_init_seed}')
        random.seed(python_init_seed)
        torch.manual_seed(seed_state['torch_seed'])
        # np.random.set_state(seed_state['numpy_seed']) # 原先固定np随机种子的方法
        np_init_seed = seed_state['numpy_init_seed'] # 现在固定np随机种子的方法
        print(f'np_init_seed: {np_init_seed}')
        np.random.seed(np_init_seed)
        # 在使用cuda时，还可以设置cuda的随机数种子
        if torch.cuda.is_available() and seed_state['torch_cuda_seed'] is not None:
            torch.cuda.manual_seed(seed_state['torch_cuda_seed'])
            torch.cuda.manual_seed_all(seed_state['torch_cuda_seed'])
        # 确保使用相同的deterministic算法以提高复现性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print(f'create new seed')
        # 获取python的随机种子
        python_init_seed = random.randrange(1000000000, 9999999999)
        print(f'python_init_seed: {python_init_seed}')
        random.seed(python_init_seed)
        # 获取np的随机种子
        # 注意前面已经通过random.seed(python_init_seed)来固定python的随机序列，
        # 此处不可用np_init_seed = random.randrange(100000, 999999)来随机，
        # 因为会使随机操作的次数与复现时不同，从而打乱固定的序列
        np_init_seed = np.random.randint(100000, 999999)
        print(f'np_init_seed: {np_init_seed}')
        np.random.seed(np_init_seed)

    if args.render_only and (args.render_test or args.render_path):
        print("======> eval <======")
        render_test(args)
    else:
        print("======> train <======")
        reconstruction(args, np_init_seed, python_init_seed)
