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
from dataset.dataset_class import DatasetRepeater
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
    parser.add_argument('--data-dir')
    parser.add_argument('--val-dir')
    parser.add_argument('--frame-shape', default=256, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--disc-learn', default=1, type=int)
    parser.add_argument('--not-bilinear', action='store_true')
    parser.add_argument('--another-resup', action='store_true')

    return parser.parse_args()


def print_fun(s):
    print(s)
    sys.stdout.flush()


def main():
    args = parse_args()
    """Create dataset and net"""
    cpu = torch.device("cpu")
    device = torch.device("cuda") if torch.cuda.is_available() else cpu
    batch_size = args.batch_size
    frame_shape = args.frame_shape
    K = args.k

    if not args.preprocessed and not args.data_dir:
        raise RuntimeError('Please provide either --preprocessed or --data-dir (path to the dataset)')
    data_dir = args.preprocessed or args.data_dir

    dataset = PreprocessDataset(K=K, path_to_preprocess=data_dir, frame_shape=frame_shape)
    dataset = DatasetRepeater(dataset, num_repeats=10 if len(dataset) < 100 else 2)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    val_loader = None
    if args.val_dir:
        val_dataset = PreprocessDataset(K=K, path_to_preprocess=args.val_dir, frame_shape=frame_shape)
        val_dataset = DatasetRepeater(val_dataset, num_repeats=10 if len(val_dataset) < 100 else 2)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    path_to_chkpt = os.path.join(args.train_dir, 'model_weights.tar')
    if os.path.isfile(path_to_chkpt):
        checkpoint = torch.load(path_to_chkpt, map_location=cpu)
        is_bilinear = checkpoint.get('is_bilinear', True)
        another_resup = checkpoint.get('another_resup', False)
    else:
        is_bilinear = not args.not_bilinear
        another_resup = args.another_resup

    G = nn.DataParallel(Generator(frame_shape, device, bilinear=is_bilinear, another_resup=another_resup).to(device))
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
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)
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
            'optimizerD': optimizerD.state_dict(),
            'is_bilinear': is_bilinear,
            'another_resup': another_resup,
        }, path_to_chkpt)
        print_fun('...Done')
        prev_step = 0
    else:
        """Loading from past checkpoint"""
        G.module.load_state_dict(checkpoint['G_state_dict'], strict=False)
        D.module.load_state_dict(checkpoint['D_state_dict'])
        try:
            optimizerG.load_state_dict(checkpoint['optimizerG'])
            optimizerD.load_state_dict(checkpoint['optimizerD'])
        except ValueError:
            pass
        prev_step = checkpoint['i_batch']

    G.train()
    D.train()

    """Training"""
    writer = tensorboardX.SummaryWriter(args.train_dir)
    num_batches = len(dataset) / args.batch_size
    log_step = int(round(0.005 * num_batches + 20))
    val_step = int(round(0.005 * num_batches + 100))
    log_epoch = 1
    if num_batches <= 100:
        log_step = 50
        log_epoch = 300 // num_batches
    else:
        val_step = 2000 // num_batches
    save_checkpoint = args.save_checkpoint
    print_fun(f"Will log each {log_step} step.")
    print_fun(f"Will save checkpoint each {save_checkpoint} step.")
    if prev_step != 0:
        print_fun(f"Starting at {prev_step} step.")

    def save_model(path):
        print_fun('Saving latest...')
        torch.save({
            'epoch': epoch,
            'G_state_dict': G.module.state_dict(),
            'D_state_dict': D.module.state_dict(),
            'num_vid': dataset.__len__(),
            'i_batch': step,
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict(),
            'is_bilinear': is_bilinear,
            'another_resup': another_resup,
        }, path)
        print_fun('Done saving latest.')

    def get_picture(tensor):
        return (tensor[0] * 127.5 + 127.5).permute([1, 2, 0]).type(torch.int32).to(cpu).numpy()

    def make_grid(tensor):
        np_image = (tensor * 127.5 + 127.5).permute([0, 2, 3, 1]).type(torch.int32).to(cpu).numpy()
        np_image = np_image.clip(0, 255).astype(np.uint8)
        canvas = np.zeros([frame_shape, frame_shape, 3])
        size = math.ceil(math.sqrt(tensor.shape[0]))
        im_size = frame_shape // size
        for i, im in enumerate(np_image):
            col = i % size
            row = i // size
            im = cv2.resize(im, (im_size, im_size))
            canvas[row * im_size:(row + 1) * im_size, col * im_size:(col + 1) * im_size] = im

        return canvas

    def run_validation():
        if val_loader is not None:
            # Validation for 1 item.
            for frames, marks, img, mark, i in val_loader:
                frames = frames.to(device).reshape([-1, *list(frames.shape[2:])])
                marks = marks.to(device).reshape([-1, *list(marks.shape[2:])])
                mark = mark.to(device)
                img = img.to(device)

                fake = G(mark, frames, marks)
                fake_score, d_fake_list = D(fake, mark)

                with torch.no_grad():
                    real_score, d_real_list = D(img, mark)

                loss_generator = loss_g(
                    img, fake, fake_score, d_fake_list, d_real_list
                )
                out1 = get_picture(fake)
                out2 = get_picture(img)
                out3 = get_picture(mark)
                out4 = make_grid(frames)

                accuracy = np.sum(np.squeeze((np.abs(out1 - out2) <= 1))) / np.prod(out1.shape)
                ssim = metrics.structural_similarity(out1.clip(0, 255).astype(np.uint8),
                                                     out2.clip(0, 255).astype(np.uint8), multichannel=True)
                print_fun(
                    f'Step {step} [{epoch}/{num_epochs}]\tVal_Loss_G: {loss_generator.item():.4f}\t'
                    f'Val_Match: {accuracy:.3f}\tVal_SSIM: {ssim:.3f}'
                )

                image = np.hstack((out1, out2, out3, out4)).clip(0, 255).astype(np.uint8)
                writer.add_image(
                    'Val_Result', image,
                    global_step=step,
                    dataformats='HWC'
                )

                writer.add_scalar('val_loss_g', loss_generator.item(), global_step=step)
                writer.add_scalar('val_match', accuracy, global_step=step)
                writer.add_scalar('val_ssim', ssim, global_step=step)
                break

    for epoch in range(0, num_epochs):
        # if epochCurrent > epoch:
        #     pbar = tqdm(dataLoader, leave=True, initial=epoch, disable=None)
        #     continue
        # Reset random generator
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
                fake = G(mark, frames, marks)
                fake_score, d_fake_list = D(fake, mark)

                with torch.no_grad():
                    real_score, d_real_list = D(img, mark)

                loss_generator = loss_g(
                    img, fake, fake_score, d_fake_list, d_real_list
                )
                loss_generator.backward(retain_graph=True)
                optimizerG.step()

            step = epoch * num_batches + i_batch + prev_step
            if step % args.disc_learn == 0:
                with torch.autograd.enable_grad():
                    optimizerG.zero_grad()
                    fake.detach_().requires_grad_()
                    optimizerD.zero_grad()
                    fake_score, d_fake_list = D(fake, mark)
                    loss_fake = loss_d_fake(fake_score)

                    real_score, d_real_list = D(img, mark)
                    loss_real = loss_d_real(real_score)

                    loss_d = loss_fake + loss_real
                    loss_d.backward()
                    optimizerD.step()

            # Output training stats
            if step % log_step == 0:
                out1 = get_picture(fake)
                out2 = get_picture(img)
                out3 = get_picture(mark)
                out4 = make_grid(frames)

                accuracy = np.sum(np.squeeze((np.abs(out1 - out2) <= 1))) / np.prod(out1.shape)
                ssim = metrics.structural_similarity(out1.clip(0, 255).astype(np.uint8), out2.clip(0, 255).astype(np.uint8), multichannel=True)
                print_fun(
                    f'Step {step} [{epoch}/{num_epochs}][{i_batch}/{len(data_loader)}]\t'
                    f'Loss_G: {loss_generator.item():.4f}\tLoss_D: {loss_d.item():.4f}\t'
                    f'Match: {accuracy:.3f}\tSSIM: {ssim:.3f}'
                )

                image = np.hstack((out1, out2, out3, out4)).clip(0, 255).astype(np.uint8)
                writer.add_image(
                    'Result', image,
                    global_step=step,
                    dataformats='HWC'
                )
                writer.add_scalar('loss_g', loss_generator.item(), global_step=step)
                writer.add_scalar('loss_d', loss_d.item(), global_step=step)
                writer.add_scalar('match', accuracy, global_step=step)
                writer.add_scalar('ssim', ssim, global_step=step)
                writer.flush()

            if step != 0 and step % save_checkpoint == 0:
                save_model(path_to_chkpt)
            if step % val_step == 0:
                run_validation()

        if epoch % log_epoch == 0:
            run_validation()
            save_model(path_to_chkpt)


if __name__ == '__main__':
    main()
