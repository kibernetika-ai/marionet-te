import argparse
import os
import glob

import torch
import cv2
import face_alignment
import numpy as np

from network.model import *
from dataset import video_extraction_conversion


"""Init"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--video', help='Path to the driver video')
    parser.add_argument('--target', help='Path to target image or video.')
    parser.add_argument('--output')
    parser.add_argument('--frame-size', type=int, default=256)

    return parser.parse_args()


def main():
    # Paths
    args = parse_args()
    frame_size = args.frame_size

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else 'cpu')
    cpu = torch.device("cpu")

    checkpoint = torch.load(args.model, map_location=cpu)

    model = Generator(
        frame_size,
        device=device,
        bilinear=checkpoint.get('is_bilinear'),
        another_resup=checkpoint.get('another_resup')
    )
    model.eval()

    """Training Init"""
    model.load_state_dict(checkpoint['G_state_dict'])
    model.to(device)

    print('Extracting target landmarks...')
    cap = cv2.VideoCapture(args.video if args.video else 0)
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D, flip_input=False, device=device.type
    )

    if args.output:
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # video_format = video.get(cv2.CAP_PROP_FORMAT)
        video_writer = cv2.VideoWriter(
            args.output, fourcc, fps,
            frameSize=(frame_size * 3, frame_size)
        )

    src_imgs, src_lmarks = extract_images(args.target, fa, image_size=frame_size)
    src_imgs = torch.from_numpy(np.array(src_imgs)).type(dtype=torch.float).permute([0, 3, 1, 2])
    src_lmarks = torch.from_numpy(np.array(src_lmarks)).type(dtype=torch.float).permute([0, 3, 1, 2])
    src_imgs = (src_imgs.to(device) - 127.5) / 127.5
    src_lmarks = (src_lmarks.to(device) - 127.5) / 127.5

    print('PRESS Q TO EXIT')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_list = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)]
        try:
            l = video_extraction_conversion.generate_landmarks(frames_list, face_aligner=fa, size=frame_size)
        except TypeError:
            continue

        x, g_y = l[0][0], l[0][1]
        x = torch.from_numpy(x.transpose([2, 0, 1])).type(dtype=torch.float)
        g_y = torch.from_numpy(g_y.transpose([2, 0, 1])).type(dtype=torch.float)
        if use_cuda:
            x, g_y = x.cuda(), g_y.cuda()

        g_y = (g_y.unsqueeze(0) - 127.5) / 127.5
        x = (x.unsqueeze(0) - 127.5) / 127.5

        x_hat = model(g_y, src_imgs, src_lmarks)

        def denorm(img):
            img = img[0].to(cpu).numpy().transpose([1, 2, 0])
            img = (img * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
            return img

        out1 = denorm(x)
        out2 = denorm(g_y)
        out3 = denorm(x_hat)

        result = cv2.cvtColor(np.hstack((out1, out2, out3)), cv2.COLOR_BGR2RGB)
        cv2.imshow('Result', result)
        if args.output:
            video_writer.write(result)

        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    if args.output:
        video_writer.release()


def extract_images(path, face_aligner, image_size=256):
    base = os.path.basename(path)
    images = []
    k = 8
    if not os.path.exists(path):
        raise RuntimeError(f'No such file or directory: {path}')

    if os.path.isfile(path):
        _, ext = os.path.splitext(base)
        if ext in {'.jpg', '.png'}:
            # Extract single image.
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for i in range(k):
                images.append(img.copy())

        elif ext in {'.mov', 'avi', 'mp4'}:
            # Extract K random frames from video.
            vc = cv2.VideoCapture(path)
            frames = vc.get(cv2.CAP_PROP_FRAME_COUNT)
            frame_indices = np.sort((np.random.randint(0, frames, size=k)))
            for idx in frame_indices:
                vc.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, img = vc.read()
                if not ret:
                    raise RuntimeError('Can not read a frame from video.')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
    else:
        # Extract K random frames from directory.
        jpg_paths = sorted(glob.glob(os.path.join(path, '*.jpg')))
        random_paths = np.random.choice(jpg_paths, size=k)
        for img_path in random_paths:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

    images_lmarks = video_extraction_conversion.generate_landmarks(images, face_aligner, size=image_size)

    return zip(*images_lmarks)


if __name__ == '__main__':
    main()

