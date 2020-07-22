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
    parser.add_argument('--target', help='Path to target image.')
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

    print('Extracting landmark...')
    img = cv2.cvtColor(cv2.imread(args.target), cv2.COLOR_BGR2RGB)
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D, flip_input=False, device=device.type
    )
    cropped_landmark, _ = video_extraction_conversion.generate_landmarks(
        [img], fa,
        size=args.frame_size,
        crop=False
    )
    cropped, landmark = cropped_landmark[0]

    src_imgs = np.expand_dims(cropped, axis=0)
    src_lmarks = np.expand_dims(landmark, axis=0)

    src_imgs = torch.from_numpy(np.array(src_imgs)).type(dtype=torch.float).permute([0, 3, 1, 2])
    src_lmarks = torch.from_numpy(np.array(src_lmarks)).type(dtype=torch.float).permute([0, 3, 1, 2])
    src_imgs = (src_imgs.to(device) - 127.5) / 127.5
    src_lmarks = (src_lmarks.to(device) - 127.5) / 127.5

    x, g_y = src_imgs, src_lmarks
    x_hat = model(src_lmarks, src_imgs, src_lmarks)

    def denorm(img):
        img = img[0].to(cpu).numpy().transpose([1, 2, 0])
        img = (img * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        return img

    out1 = denorm(x)
    out2 = denorm(g_y)
    out3 = denorm(x_hat)

    result = cv2.cvtColor(np.hstack((out1, out2, out3)), cv2.COLOR_BGR2RGB)
    cv2.imshow('Result', result)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()

