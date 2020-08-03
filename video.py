import argparse
import os
import glob

import torch
import cv2
import face_alignment
from ml_serving.drivers import driver
import numpy as np

from network.model import *
from dataset import video_extraction_conversion
from utils import utils


"""Init"""
face_model_path = (
    '/opt/intel/openvino/deployment_tools/intel_models'
    '/face-detection-adas-0001/FP32/face-detection-adas-0001.xml'
)


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
    face_driver = driver.load_driver('openvino')().load_model(face_model_path)

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

    src_imgs, src_lmarks, av_margin = extract_images(args.target, face_driver, fa, image_size=frame_size)
    # src_imgs = torch.from_numpy(np.array(src_imgs)).type(dtype=torch.float).permute([0, 3, 1, 2])
    # src_lmarks = torch.from_numpy(np.array(src_lmarks)).type(dtype=torch.float).permute([0, 3, 1, 2])
    src_imgs = src_imgs.to(device)
    src_lmarks = src_lmarks.to(device)

    print('PRESS Q TO EXIT')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        norm_image, norm_mark = preprocess_image(frame, fa, face_driver, frame_size)
        if norm_image is None:
            continue

        norm_mark = norm_mark.to(device)

        # x, g_y = l[0][0], l[0][1]
        # x = torch.from_numpy(x.transpose([2, 0, 1])).type(dtype=torch.float)
        # g_y = torch.from_numpy(g_y.transpose([2, 0, 1])).type(dtype=torch.float)
        # if use_cuda:
        #     x, g_y = x.cuda(), g_y.cuda()
        #
        # g_y = (g_y.unsqueeze(0) - 127.5) / 127.5
        # x = (x.unsqueeze(0) - 127.5) / 127.5

        x_hat = model(norm_mark, src_imgs, src_lmarks)

        def denorm(img):
            img = img[0].to(cpu).numpy().transpose([1, 2, 0])
            img = (img * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
            return img

        out1 = denorm(norm_image)
        out2 = denorm(norm_mark)
        out3 = denorm(x_hat)
        out4 = denorm(src_imgs)

        result = cv2.cvtColor(np.hstack((out1, out2, out3, out4)), cv2.COLOR_BGR2RGB)
        cv2.imshow('Result', result)
        if args.output:
            video_writer.write(result)

        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    if args.output:
        video_writer.release()


def preprocess_image(image, face_aligner, face_driver, image_size=256):
    boxes = utils.get_boxes(face_driver, image[:, :, ::-1])
    if len(boxes) != 1:
        return None, None
    margin = 0.4
    crop_box = utils.get_crop_box(image, boxes[0], margin=margin)
    cropped = utils.crop_by_box(image, boxes[0], margin=margin)
    # cv2.imshow('Cropped', cropped)
    # cv2.waitKey(1)
    resized = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_AREA)

    landmarks = face_aligner.get_landmarks_from_image(image, [crop_box])[0]
    landmarks -= [crop_box[0], crop_box[1]]
    x_factor, y_factor = (crop_box[2] - crop_box[0]) / image_size, (crop_box[3] - crop_box[1]) / image_size
    landmarks /= [x_factor, y_factor]
    landmark_img = video_extraction_conversion.draw_landmark(
        landmarks, size=(image_size, image_size, 3)
    )

    norm_image = torch.from_numpy(np.expand_dims(resized, axis=0)).type(dtype=torch.float)  # K,256,256,3
    norm_mark = torch.from_numpy(np.expand_dims(landmark_img, axis=0)).type(dtype=torch.float)  # K,256,256,3
    norm_image = (norm_image.permute([0, 3, 1, 2]) - 127.5) / 127.5
    norm_mark = (norm_mark.permute([0, 3, 1, 2]) - 127.5) / 127.5  # K,3,256,256
    return norm_image, norm_mark


def extract_images(path, face_driver, face_aligner, image_size=256):
    base = os.path.basename(path)
    images = []
    k = 1
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

    imgs = []
    lmarks = []
    for image in images:
        norm_image, norm_mark = preprocess_image(image, face_aligner, face_driver, image_size)
        if norm_image is None:
            continue

        imgs.append(norm_image)
        lmarks.append(norm_mark)
    # images_lmarks, av_margin = video_extraction_conversion.generate_landmarks(
    #     images, face_aligner, size=image_size, crop=False, margins=[0.4, 0.4, 0.4, 0.4]
    # )
    # imgs, lmarks = zip(*images_lmarks)

    return torch.stack(imgs).squeeze(dim=1), torch.stack(lmarks).squeeze(dim=1), [0.4, 0.4, 0.4, 0.4]


if __name__ == '__main__':
    main()

