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

from dataset.dataset_class import LandmarkDataset
from dataset.dataset_class import DatasetRepeater
from dataset.video_extraction_conversion import *
from loss.loss_generator import *
from network import disentangler
from network.resblocks import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--save-checkpoint', type=int, default=1000)
    parser.add_argument('--train-dir', default='train')
    parser.add_argument('--data-dir')
    parser.add_argument('--frame-shape', default=256, type=int)
    parser.add_argument('--workers', default=4, type=int)

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

    dataset = LandmarkDataset(root_dir=args.data_dir, frame_shape=frame_shape)
    dataset = DatasetRepeater(dataset, num_repeats=100 if len(dataset) < 100 else 20)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    path_to_chkpt = os.path.join(args.train_dir, 'model_weights.tar')
    if os.path.isfile(path_to_chkpt):
        checkpoint = torch.load(path_to_chkpt, map_location=cpu)
        if checkpoint.get('mean_landmark'):
            dataset.dataset.set_mean_landmark(checkpoint['mean_landmark'])

    d = nn.DataParallel(disentangler.Disentangler().to(device))

    optimizer = optim.Adam(
        params=list(d.parameters()),
        lr=5e-5,
        amsgrad=False
    )
    """Criterion"""

    """Training init"""
    epoch = i_batch = 0
    num_epochs = args.epochs

    # initiate checkpoint if unexistent
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)

    if not os.path.isfile(path_to_chkpt):
        def init_weights(m):
            if type(m) == nn.Conv2d:
                torch.nn.init.xavier_uniform(m.weight)

        d.apply(init_weights)

        print_fun('Initiating new checkpoint...')
        torch.save({
            'epoch': epoch,
            'state_dict': d.module.state_dict(),
            'num_vid': dataset.__len__(),
            'i_batch': i_batch,
            'optimizer': optimizer.state_dict(),
            'mean_landmark': dataset.dataset.get_mean_landmark(),
        }, path_to_chkpt)
        print_fun('...Done')
        prev_step = 0
    else:
        """Loading from past checkpoint"""
        d.module.load_state_dict(checkpoint['state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        prev_step = checkpoint['i_batch']

    d.train()

    """Training"""
    writer = tensorboardX.SummaryWriter(args.train_dir)
    num_batches = len(dataset) / args.batch_size
    log_step = int(round(0.005 * num_batches + 20))
    log_epoch = 1
    if num_batches <= 100:
        log_step = 50
        log_epoch = 300 // num_batches
    save_checkpoint = args.save_checkpoint
    print_fun(f"Will log each {log_step} step.")
    print_fun(f"Will save checkpoint each {save_checkpoint} step.")
    if prev_step != 0:
        print_fun(f"Starting at {prev_step} step.")

    for epoch in range(0, num_epochs):
        # Reset random generator
        np.random.seed(int(time.time()))
        for i_batch, (frames, marks, img, mark, i) in enumerate(data_loader):

            frames = frames.to(device).reshape([-1, *list(frames.shape[2:])])
            marks = marks.to(device).reshape([-1, *list(marks.shape[2:])])
            mark = mark.to(device)
            img = img.to(device)

            with torch.autograd.enable_grad():
                # zero the parameter gradients
                optimizer.zero_grad()

                fake = d(mark, frames, marks)

                loss_generator = loss_g(
                    img, fake, fake_score, d_fake_list, d_real_list
                )
                loss_generator.backward()
                optimizer.step()

            step = epoch * num_batches + i_batch + prev_step

            # Output training stats
            if step % log_step == 0:
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
                        canvas[row * im_size:(row+1) * im_size, col*im_size:(col+1) * im_size] = im

                    return canvas

                out1 = get_picture(fake)
                out2 = get_picture(img)
                out3 = get_picture(mark)
                out4 = make_grid(frames)

                accuracy = np.sum(np.squeeze((np.abs(out1 - out2) <= 1))) / np.prod(out1.shape)
                ssim = metrics.structural_similarity(out1.clip(0, 255).astype(np.uint8), out2.clip(0, 255).astype(np.uint8), multichannel=True)
                # print_fun(
                #     'Step %d [%d/%d][%d/%d]\tLoss_G: %.4f\tLoss_D: %.4f\tMatch: %.3f\tSSIM: %.3f'
                #     % (step, epoch, num_epochs, i_batch, len(data_loader),
                #        loss_generator.item(), loss_d.item(), accuracy, ssim)
                # )

                image = np.hstack((out1, out2, out3, out4)).clip(0, 255).astype(np.uint8)
                writer.add_image(
                    'Result', image,
                    global_step=step,
                    dataformats='HWC'
                )
                writer.add_scalar('loss_g', loss_generator.item(), global_step=step)
                writer.add_scalar('match', accuracy, global_step=step)
                writer.add_scalar('ssim', ssim, global_step=step)
                writer.flush()

            if step != 0 and step % save_checkpoint == 0:
                print_fun('Saving latest...')
                torch.save({
                    'epoch': epoch,
                    'state_dict': d.module.state_dict(),
                    'num_vid': dataset.__len__(),
                    'i_batch': step,
                    'optimizer': optimizer.state_dict(),
                    'mean_landmark': dataset.dataset.get_mean_landmark(),
                },
                    path_to_chkpt
                )

        if epoch % log_epoch == 0:
            print_fun('Saving latest...')
            torch.save({
                'epoch': epoch,
                'state_dict': d.module.state_dict(),
                'num_vid': dataset.__len__(),
                'i_batch': step,
                'optimizer': optimizer.state_dict(),
                'mean_landmark': dataset.dataset.get_mean_landmark(),
            },
                path_to_chkpt
            )
            print_fun('...Done saving latest')


if __name__ == '__main__':
    main()
