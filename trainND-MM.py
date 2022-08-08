import argparse
import os
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import random
import torch.nn.functional as F
from config import  IMAGENET_Train_PATH
from custom_loss import NegativeCrossEntropy, BoundedLogitLossFixedRef
import time
import math

MODE = "bilinear"
import pandas as pd


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


def ND_loss(logits_student, logits_teacher, target, alpha=0.5, beta=0.5, temperature=1):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, size_average=False)
            * (temperature ** 2)
            / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
            * (temperature ** 2)
            / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


def project_simplex(v, z=1.0, axis=-1):
    def _project_simplex_2d(v, z):
        """
        Helper function, assuming that all vectors are arranged in rows of v.
        :param v: NxD torch tensor; Duchi et al. algorithm is applied to each row in vecotrized form
        :param z: Vectors will be projected onto the z-Simplex: \sum w_i = z.
        :return: w: result of the projection
        """
        with torch.no_grad():
            shape = v.shape
            if shape[1] == 1:
                w = v.clone().detach()
                w[:] = z
                return w

            mu = torch.sort(v, dim=1)[0]
            mu = torch.flip(mu, dims=(1,))
            cum_sum = torch.cumsum(mu, dim=1)
            j = torch.unsqueeze(torch.arange(1, shape[1] + 1, dtype=mu.dtype, device=mu.device), 0)
            rho = torch.sum(mu * j - cum_sum + z > 0.0, dim=1, keepdim=True) - 1
            max_nn = cum_sum[torch.arange(shape[0]), rho[:, 0]]
            theta = (torch.unsqueeze(max_nn, -1) - z) / (rho.type(max_nn.dtype) + 1)
            w = torch.clamp(v - theta, min=0.0)
            return w

    with torch.no_grad():
        shape = v.shape

        if len(shape) == 1:
            return _project_simplex_2d(torch.unsqueeze(v, 0), z)[0, :]
        else:
            axis = axis % len(shape)
            t_shape = tuple(range(axis)) + tuple(range(axis + 1, len(shape))) + (axis,)
            tt_shape = tuple(range(axis)) + (len(shape) - 1,) + tuple(range(axis, len(shape) - 1))
            v_t = v.permute(t_shape)
            v_t_shape = v_t.shape
            v_t_unroll = torch.reshape(v_t, (-1, v_t_shape[-1]))

            w_t = _project_simplex_2d(v_t_unroll, z)

            w_t_reroll = torch.reshape(w_t, v_t_shape)
            return w_t_reroll.permute(tt_shape)


def normalize(x):
    """
    Normalizes a batch of images with size (batch_size, 3, height, width)
    by mean and std dev expected by PyTorch models
    """
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])
    return (x - mean.type_as(x)[None, :, None, None]) / std.type_as(x)[None, :, None, None]


def parse_arguments():
    parser = argparse.ArgumentParser(description='Transferable Targeted Perturbations')
    parser.add_argument('--match_target', type=int, default=24, help='Target Domain samples')
    parser.add_argument('--lr_w', type=float, default=0.003, help='learning rate of W')
    parser.add_argument('--lam', type=float, default=1, help='learning rate of W')
    parser.add_argument('--batch_size', type=int, default=20, help='Number of trainig samples/batch')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--eps', type=int, default=10, help='Perturbation Budget during training, eps')
    parser.add_argument('--save_dir', type=str, default='result', help='Directory to save generators')
    parser.add_argument('--ngpu', type=int, default=1,
                        help='Number of used GPUs (0 = CPU) (default: 1)')
    parser.add_argument('--confidence', default=10., type=float,
                        help='Confidence value for C&W losses (default: 0.0)')
    parser.add_argument('--pretrained_seed', type=int, default=0,
                        help='Seed used in the generation process (default: 0)')

    args = parser.parse_args()
    print(args)
    if args.pretrained_seed is None:
        args.pretrained_seed = random.randint(1, 10000)

    args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()
    return args


def train(train_loader, model_pool, noise, epoch, optimizer, criterion, args, W):
    noise.requires_grad = True
    W.requires_grad = True
    optimizer.zero_grad()
    K = len(model_pool)
    for i, (input, _) in enumerate(train_loader):
        input = input.cuda()
        if args.match_target != -1:
            target = torch.ones(input.shape[0], dtype=torch.int64) * args.match_target
            target = target.cuda()
        # generate adversarial examples
        current_noise = noise
        current_noise = F.interpolate(
            current_noise.unsqueeze(0),
            mode=MODE, size=tuple(input.shape[-2:]),
            align_corners=True,
        ).squeeze()
        adv = torch.clamp(input + current_noise, 0, 1)
        optimizer.zero_grad()
        # Fix W , optimizing uap
        adv_logits = 0
        logits_oig = 0
        logits_set = []
        for j, net in enumerate(model_pool):
            ll = net(normalize(adv))
            logits_set.append(ll)
            adv_logits += W[j] * ll
            if args.match_target == -1:
                logits_oig += W[j] * net(normalize(input)).detach()

        if args.match_target == -1:
            loss_fuse = criterion(logits_oig, adv_logits)
        else:
            loss_fuse = criterion(adv_logits, target.cuda())


        # Add kl
        loss_kl = 0
        for ll in logits_set:
            if args.match_target != -1:
                loss_kl += ND_loss(ll, adv_logits, target.cuda(), alpha=0, beta=1, temperature=10)
            else:
                loss_kl += ND_loss(ll, adv_logits, adv_logits.argmax(dim=-1).detach(), alpha=0, beta=1, temperature=10)
        loss = loss_fuse + args.lam*loss_kl

        loss.backward()
        optimizer.step()
        noise.data = torch.clamp(noise.data, -(args.eps / 255), args.eps / 255)

        ## Fix uap, maxmize W
        update_noise = F.interpolate(
            noise.unsqueeze(0),
            mode=MODE, size=tuple(input.shape[-2:]),
            align_corners=True,
        ).squeeze()
        update_adv = torch.clamp(input + update_noise, 0, 1)

        W.requires_grad = True
        adv_logits_w = 0
        for j, net in enumerate(model_pool):
            logit = net(normalize(update_adv))
            adv_logits_w += W[j] * logit

        if args.match_target == -1:
            loss_w = criterion(logits_oig, adv_logits_w) - K * (W - 1 / K).norm()
        else:
            loss_w = criterion(adv_logits_w, target) - K * (W - 1 / K).norm()

        grad_w = torch.autograd.grad(loss_w, W, retain_graph=False, create_graph=False)[0]
        W.data += args.lr_w * grad_w

        # Constrain W
        W = project_simplex(W)
        if i % 50 == 0:
            print(W)
            print('>> Train: [{0}][{1}/{2}]\t'
                  "Loss {loss:.4f}\t"
                  "Noise l2: {noise:.4f}".format(
                epoch + 1,
                i,
                len(train_loader),
                loss=loss.item(),
                noise=noise.norm())
            )
    noise.requires_grad = False
    return noise, W




def main():
    args = parse_arguments()

    random.seed(args.pretrained_seed)
    np.random.seed(args.pretrained_seed)
    torch.manual_seed(args.pretrained_seed)
    torch.cuda.manual_seed(args.pretrained_seed)
    torch.cuda.manual_seed_all(args.pretrained_seed)

    #### save_dir path of trained uap
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model_1 = torchvision.models.vgg16(pretrained=True).eval()
    model_2 = torchvision.models.resnet50(pretrained=True).eval()
    model_3 = torchvision.models.densenet121(pretrained=True).eval()

    if torch.cuda.device_count() > 1:
        model_1 = torch.nn.DataParallel(model_1, device_ids=list(range(2)))
        model_2 = torch.nn.DataParallel(model_2, device_ids=list(range(2)))
        model_3 = torch.nn.DataParallel(model_3, device_ids=list(range(2)))

    model_1 = model_1.cuda()
    model_2 = model_2.cuda()
    model_3 = model_3.cuda()
    model_pool = [model_1, model_2, model_3]

    # universal noise
    scale_size = 256
    img_size = 224
    noise = torch.zeros((3, img_size, img_size)).cuda()
    noise.requires_grad = True

    optimizer = optim.Adam([noise], lr=0.005)

    if args.match_target == -1:
        criterion = NegativeCrossEntropy()
    else:
        criterion = BoundedLogitLossFixedRef(num_classes=1000, confidence=args.confidence,
                                             use_cuda=args.use_cuda)
    ## data loader##
    train_transform = transforms.Compose([
        transforms.Resize(scale_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    train_set = torchvision.datasets.ImageFolder(IMAGENET_Train_PATH,
                                                 train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                               pin_memory=True)
    train_size = len(train_set)
    print('Training data size:', train_size)

    W = torch.ones(len(model_pool)).cuda() / len(model_pool)
    for epoch in range(args.epochs):
        noise, W = train(train_loader, model_pool, noise, epoch, optimizer, criterion, args, W)

    np.save(args.save_dir + "/mim_tg_{}".format(args.match_target), noise.cpu().data.numpy())


if __name__ == '__main__':
    main()






