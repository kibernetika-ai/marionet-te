import glob
import time

import torch
from torch.utils.data import Dataset
import face_alignment

from .video_extraction_conversion import *


class VidDataSet(Dataset):
    def __init__(self, K, path_to_mp4, device, size=256):
        self.K = K
        self.size = size
        self.path_to_mp4 = path_to_mp4
        self.device = device
        self.face_aligner = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D,
            flip_input=False,
            device=device
        )
        self.video_paths = glob.glob(os.path.join(path_to_mp4, '*/*/*.mp4'))

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        vid_idx = idx
        path = self.video_paths[vid_idx]
        ok = False
        while not ok:
            try:
                frame_mark = select_frames(path, self.K)
                frame_mark = generate_landmarks(frame_mark, self.face_aligner, size=self.size)
                ok = True
            except Exception:
                vid_idx = torch.randint(low=0, high=len(self.video_paths), size=(1,))[0].item()
                path = self.video_paths[vid_idx]
        frame_mark = torch.from_numpy(np.array(frame_mark)).type(dtype=torch.float)  # K,2,224,224,3
        frame_mark = frame_mark.permute([0, 1, 4, 2, 3]) / 255.  # K,2,3,224,224

        g_idx = torch.randint(low=0, high=self.K, size=(1, 1))
        x = frame_mark[g_idx, 0].squeeze()
        g_y = frame_mark[g_idx, 1].squeeze()

        return frame_mark, x, g_y, vid_idx


class PreprocessDataset(Dataset):
    def __init__(self, K, path_to_preprocess, frame_shape=224):
        self.K = K
        self.path_to_preprocess = path_to_preprocess
        self.frame_shape = frame_shape

        self.video_dirs = glob.glob(os.path.join(path_to_preprocess, '*/*'))

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        vid_idx = idx
        video_dir = self.video_dirs[vid_idx]
        lm_path = os.path.join(video_dir, 'landmarks.npy')
        jpg_paths = sorted(glob.glob(os.path.join(video_dir, '*.jpg')))
        if not jpg_paths:
            raise RuntimeError('Dataset does not contain .jpg files.')
        if os.path.exists(lm_path):
            all_landmarks = np.load(lm_path)

        while not os.path.exists(lm_path) or len(all_landmarks) != len(jpg_paths):
            vid_idx = vid_idx // 2
            video_dir = self.video_dirs[vid_idx]
            lm_path = os.path.join(video_dir, 'landmarks.npy')
            if not os.path.exists(lm_path):
                continue
            jpg_paths = glob.glob(os.path.join(video_dir, '*.jpg'))
            all_landmarks = np.load(lm_path)
            if len(all_landmarks) != len(jpg_paths):
                continue

        # Select K paths
        random_indices = np.random.randint(0, len(jpg_paths), size=(self.K,))
        paths = np.array(jpg_paths)[random_indices]
        landmarks = all_landmarks[random_indices]

        frames = []
        marks = []
        for i, path in enumerate(paths):
            frame = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            cur_landmark = landmarks[i].copy()
            if frame.shape[:2] != (self.frame_shape, self.frame_shape):
                x_factor, y_factor = frame.shape[1] / self.frame_shape, frame.shape[0] / self.frame_shape
                frame = cv2.resize(frame, (self.frame_shape, self.frame_shape), interpolation=cv2.INTER_AREA)
                cur_landmark /= [x_factor, y_factor]
            lmark = draw_landmark(cur_landmark, size=frame.shape)
            # cv2.imshow('img', np.hstack((frame, lmark))[:, :, ::-1])
            # cv2.waitKey(0)
            # exit()
            frames.append(frame)
            marks.append(lmark)

        frames = torch.from_numpy(np.array(frames)).type(dtype=torch.float)  # K,224,224,3
        marks = torch.from_numpy(np.array(marks)).type(dtype=torch.float)  # K,224,224,3
        frames = frames.permute([0, 3, 1, 2]) / 255.  # K,3,224,224
        marks = marks.permute([0, 3, 1, 2]) / 255.  # K,3,224,224
        # frame_mark = frame_mark.requires_grad_(False)

        g_idx = np.random.randint(low=0, high=self.K)
        img = frames[g_idx]
        mark = marks[g_idx]

        return frames, marks, img, mark, vid_idx
