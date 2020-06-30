"""Main"""
import argparse
import sys
import time

import matplotlib
from skimage import metrics
import tensorboardX
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.dataset_class import PreprocessDataset
from dataset.dataset_class import VidDataSet
from dataset.video_extraction_conversion import *
from loss.loss_discriminator import *
from loss.loss_generator import *
from network.model import *
from network.resblocks import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', default=8, type=int)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--preprocessed')
    parser.add_argument('--save-checkpoint', type=int, default=1000)
    parser.add_argument('--train-dir', default='train')
    parser.add_argument('--vggface-dir', default='.')
    parser.add_argument('--data-dir', default='../image2image/ds_fa_vox')
    parser.add_argument('--frame-shape', default=256, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--fa-device', default='cuda:0' if torch.cuda.is_available() else 'cpu')

    return parser.parse_args()


def print_fun(s):
    print(s)
    sys.stdout.flush()


def main():
    args = parse_args()
    """Create dataset and net"""
    matplotlib.use('agg')
    cpu = torch.device("cpu")
    device = torch.device("cuda") if torch.cuda.is_available() else cpu
    batch_size = args.batch_size
    frame_shape = args.frame_shape
    K = args.k

    if args.preprocessed:
        dataset = PreprocessDataset(K=K, path_to_preprocess=args.preprocessed, frame_shape=frame_shape)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.workers,
        )
    else:
        dataset = VidDataSet(
            K=K, path_to_mp4=args.data_dir,
            device=args.fa_device, size=frame_shape
        )
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.workers if 'cuda' not in args.fa_device else 0,
        )

    path_to_chkpt = os.path.join(args.train_dir, 'model_weights.tar')

    G = nn.DataParallel(Generator(frame_shape).to(device))
    D = nn.DataParallel(SNResNetProjectionDiscriminator().to(device))

    G.train()
    D.train()

    optimizerG = optim.Adam(
        params=list(G.parameters()),
        lr=5e-5,
        amsgrad=False
    )
    optimizerD = optim.Adam(
        params=D.parameters(),
        lr=2e-4,
        amsgrad=False)

    """Criterion"""
    loss_d_real = LossDSCreal()
    loss_d_fake = LossDSCfake()
    loss_g = LossG(
        os.path.join(args.vggface_dir, 'Pytorch_VGGFACE_IR.py'),
        os.path.join(args.vggface_dir, 'Pytorch_VGGFACE.pth'),
        device
    )

    """Training init"""
    epoch = i_batch = 0

    num_epochs = args.epochs

    # initiate checkpoint if inexistant
    if not os.path.isfile(path_to_chkpt):
        def init_weights(m):
            if type(m) == nn.Conv2d:
                torch.nn.init.xavier_uniform(m.weight)

        G.apply(init_weights)
        D.apply(init_weights)

        print_fun('Initiating new checkpoint...')
        torch.save({
            'epoch': epoch,
            'G_state_dict': G.module.state_dict(),
            'D_state_dict': D.module.state_dict(),
            'num_vid': dataset.__len__(),
            'i_batch': i_batch,
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict()
        }, path_to_chkpt)
        print_fun('...Done')

    """Loading from past checkpoint"""
    checkpoint = torch.load(path_to_chkpt, map_location=cpu)
    G.module.load_state_dict(checkpoint['G_state_dict'], strict=False)
    D.module.load_state_dict(checkpoint['D_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG'])
    optimizerD.load_state_dict(checkpoint['optimizerD'])
    prev_step = checkpoint['i_batch']

    G.train()
    D.train()

    """Training"""
    writer = tensorboardX.SummaryWriter(args.train_dir)
    num_batches = len(dataset) / args.batch_size
    log_step = int(round(0.005 * num_batches + 20))
    log_epoch = 1
    if num_batches <= 10:
        log_step = 50
        log_epoch = 100 // num_batches
    save_checkpoint = args.save_checkpoint
    print_fun(f"Will log each {log_step} step.")
    print_fun(f"Will save checkpoint each {save_checkpoint} step.")
    if prev_step != 0:
        print_fun(f"Starting at {prev_step} step.")

    for epoch in range(0, num_epochs):
        # if epochCurrent > epoch:
        #     pbar = tqdm(dataLoader, leave=True, initial=epoch, disable=None)
        #     continue
        # Reset random generator
        np.random.seed(int(time.time()))
        for i_batch, (frames, marks, img, mark, i) in enumerate(data_loader):

            frames = frames.to(device).reshape([-1, *list(frames.shape[2:])])
            marks = marks.to(device).reshape([-1, *list(marks.shape[2:])])
            mark = mark.to(device)
            img = img.to(device)

            with torch.autograd.enable_grad():
                # zero the parameter gradients
                optimizerG.zero_grad()
                optimizerD.zero_grad()

                # train G and D
                fake = G(img, frames, marks)
                fake_score, d_fake_list = D(fake, mark)
                with torch.no_grad():
                    real_score, d_real_list = D(img, mark)

                loss_generator = loss_g(
                    img, fake, fake_score, real_score, d_fake_list, d_real_list
                )
                loss_generator.backward(retain_graph=False)
                optimizerG.step()

            with torch.autograd.enable_grad():
                optimizerG.zero_grad()
                optimizerD.zero_grad()
                fake.detach_().requires_grad_()
                fake_score, d_fake_list = D(fake, mark)
                loss_fake = loss_d_fake(fake_score)

                real_score, d_real_list = D(img, mark)
                loss_real = loss_d_real(real_score)

                loss_d = loss_fake + loss_real
                loss_d.backward(retain_graph=False)
                optimizerD.step()

            step = epoch * num_batches + i_batch + prev_step
            # Output training stats
            if step % log_step == 0:
                out = (fake[0] * 255).permute([1, 2, 0])
                out1 = out.type(torch.int32).to(cpu).numpy()

                out = (img[0] * 255).permute([1, 2, 0])
                out2 = out.type(torch.int32).to(cpu).numpy()

                out = (mark[0] * 255).permute([1, 2, 0])
                out3 = out.type(torch.int32).to(cpu).numpy()
                accuracy = np.sum(np.squeeze((np.abs(out1 - out2) <= 1))) / np.prod(out.shape)
                ssim = metrics.structural_similarity(out1.astype(np.uint8).clip(0, 255), out2.astype(np.uint8).clip(0, 255), multichannel=True)
                print_fun(
                    'Step %d [%d/%d][%d/%d]\tLoss_D: %.4f\tMatch: %.3f\tSSIM: %.3f'
                    % (step, epoch, num_epochs, i_batch, len(data_loader),
                       loss_d.item(), accuracy, ssim)
                )

                image = np.hstack((out1, out2, out3)).astype(np.uint8).clip(0, 255)
                writer.add_image(
                    'Result', image,
                    global_step=step,
                    dataformats='HWC'
                )
                # writer.add_scalar('loss_g', lossG.item(), global_step=step)
                writer.add_scalar('loss_d', loss_d.item(), global_step=step)
                writer.add_scalar('match', accuracy, global_step=step)
                writer.add_scalar('ssim', ssim, global_step=step)
                writer.flush()

            if step != 0 and step % save_checkpoint == 0:
                print_fun('Saving latest...')
                torch.save({
                    'epoch': epoch,
                    'G_state_dict': G.module.state_dict(),
                    'D_state_dict': D.module.state_dict(),
                    'num_vid': dataset.__len__(),
                    'i_batch': step,
                    'optimizerG': optimizerG.state_dict(),
                    'optimizerD': optimizerD.state_dict()
                },
                    path_to_chkpt
                )

        if epoch % log_epoch == 0:
            print_fun('Saving latest...')
            torch.save({
                'epoch': epoch,
                'G_state_dict': G.module.state_dict(),
                'D_state_dict': D.module.state_dict(),
                'num_vid': dataset.__len__(),
                'i_batch': step,
                'optimizerG': optimizerG.state_dict(),
                'optimizerD': optimizerD.state_dict()
            },
                path_to_chkpt
            )
            print_fun('...Done saving latest')


if __name__ == '__main__':
    main()
