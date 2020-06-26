import cv2
import random
import numpy as np
import os


def select_frames(video_path, K):
    cap = cv2.VideoCapture(video_path)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # unused
    # w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rand_frames_idx = []
    for i in range(K):
        idx = random.randint(0, n_frames - 1)
        rand_frames_idx.append(idx)

    rand_frames_idx = sorted(rand_frames_idx)
    frames_list = []

    # Read until video is completed or no frames needed
    for frame_idx in rand_frames_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    return frames_list


def draw_landmark(landmark, canvas=None, size=None):
    if canvas is None:
        canvas = (np.zeros(size)).astype(np.uint8)

    colors = [
        (0, 255, 0),
        (255, 255, 0),
        (255, 255, 0),
        (255, 0, 0),
        (255, 0, 0),
        (0, 0, 255),
        (0, 0, 255),
        (0, 255, 255),
        (0, 255, 255),
    ]

    chin = landmark[0:17]
    left_brow = landmark[17:22]
    right_brow = landmark[22:27]
    left_eye = landmark[36:42]
    left_eye = np.concatenate((left_eye, [landmark[36]]))
    right_eye = landmark[42:48]
    right_eye = np.concatenate((right_eye, [landmark[42]]))
    nose1 = landmark[27:31]
    # nose1 = np.concatenate((nose1, [landmark[33]]))
    nose2 = landmark[31:36]
    mouth = landmark[48:60]
    mouth = np.concatenate((mouth, [landmark[48]]))
    mouth_internal = landmark[60:68]
    mouth_internal = np.concatenate((mouth_internal, [landmark[60]]))
    lines = np.array([
        chin, left_brow, right_brow,
        left_eye, right_eye, nose1, nose2,
        mouth_internal,
        mouth,
    ])
    for i, line in enumerate(lines):
        cur_color = colors[i]
        cv2.polylines(
            canvas,
            np.int32([line]), False,
            cur_color, thickness=1, lineType=cv2.LINE_AA
        )

    return canvas


def generate_landmarks(frames_list, face_aligner, size=256):
    frame_landmark_list = []
    fa = face_aligner

    for i in range(len(frames_list)):
        # try:
        input = frames_list[i]
        preds = fa.get_landmarks(input)[0]

        # crop frame
        maxx, maxy = np.max(preds, axis=0)
        minx, miny = np.min(preds, axis=0)
        margin = 0.4
        margin_top = margin + 0.3
        miny = max(int(miny - (maxy - miny) * margin_top), 0)
        maxy = min(int(maxy + (maxy - miny) * margin), input.shape[0])
        minx = max(int(minx - (maxx - minx) * margin), 0)
        maxx = min(int(maxx + (maxx - minx) * margin), input.shape[1])
        input = input[miny:maxy, minx:maxx]
        preds -= [minx, miny]

        if input.shape[:2] != (size, size):
            x_factor, y_factor = input.shape[1] / size, input.shape[0] / size
            input = cv2.resize(input, (size, size), interpolation=cv2.INTER_AREA)
            preds /= [x_factor, y_factor]

        data = draw_landmark(preds, size=input.shape)

        # if resize:
        #     input = cv2.resize(input, (size, size), interpolation=cv2.INTER_AREA)
        #     data = cv2.resize(data, (size, size), interpolation=cv2.INTER_AREA)
        frame_landmark_list.append((input, data))

    for i in range(len(frames_list) - len(frame_landmark_list)):
        # filling frame_landmark_list in case of error
        frame_landmark_list.append(frame_landmark_list[i])

    return frame_landmark_list


def select_images_frames(path_to_images):
    images_list = []
    for image_name in os.listdir(path_to_images):
        img = cv2.imread(os.path.join(path_to_images, image_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images_list.append(img)
    return images_list
