import glob
import time

import torch
from torch.utils.data import Dataset
import face_alignment

from .video_extraction_conversion import *
from utils import utils


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


class LandmarkDataset(Dataset):
    def __init__(self, root_dir, frame_shape=256):
        self.path_to_preprocess = root_dir
        self.frame_shape = frame_shape

        self.video_dirs = glob.glob(os.path.join(root_dir, '*/*'))
        self.mean_landmark = None

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        vid_idx = idx
        video_dir = self.video_dirs[vid_idx]
        lm_path = os.path.join(video_dir, 'landmarks.npy')
        jpg_paths = sorted(glob.glob(os.path.join(video_dir, '*.jpg')))
        # if not jpg_paths:
        #     raise RuntimeError('Dataset does not contain .jpg files.')
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
        random_indices = np.random.randint(0, len(jpg_paths))
        path = np.array(jpg_paths)[random_indices]
        landmark = all_landmarks[random_indices]
        mean_identity_landmark = all_landmarks.mean(axis=0) / self.frame_shape

        frame = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        if frame.shape[:2] != (self.frame_shape, self.frame_shape):
            x_factor, y_factor = frame.shape[1] / self.frame_shape, frame.shape[0] / self.frame_shape
            frame = cv2.resize(frame, (self.frame_shape, self.frame_shape), interpolation=cv2.INTER_AREA)
            landmark /= [x_factor, y_factor]
            # cv2.imshow('img', np.hstack((frame, lmark))[:, :, ::-1])
            # cv2.waitKey(0)
            # exit()

        frame = (frame.permute([2, 0, 1]) - 127.5) / 127.5

        id_landmark = mean_identity_landmark - self.get_mean_landmark()
        return frame, landmark, id_landmark

    def set_mean_landmark(self, mean_landmark):
        self.mean_landmark = mean_landmark

    def get_mean_landmark(self):
        if self.mean_landmark is not None:
            return self.mean_landmark

        utils.print_fun('Computing mean landmark...')

        all_landmarks = None
        for vid_dir in self.video_dirs:
            lm_path = os.path.join(vid_dir, 'landmarks.npy')
            if os.path.exists(lm_path):
                landmarks = np.load(lm_path)
            else:
                continue

            if all_landmarks is None:
                all_landmarks = landmarks
            else:
                all_landmarks = np.concatenate((all_landmarks, landmarks))

        self.mean_landmark = all_landmarks.mean(axis=0) / self.frame_shape

        utils.print_fun('Done computing mean landmark.')
        return self.mean_landmark


class PreprocessDataset(Dataset):
    def __init__(self, K, path_to_preprocess, frame_shape=224):
        self.K = K
        self.path_to_preprocess = path_to_preprocess
        self.frame_shape = frame_shape

        self.video_dirs = glob.glob(os.path.join(path_to_preprocess, '*/*'))
        self.mean_landmark = None

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        np.random.seed(int(time.time()))

        vid_idx = idx
        video_dir = self.video_dirs[vid_idx]
        lm_path = os.path.join(video_dir, 'landmarks.npy')
        jpg_paths = sorted(glob.glob(os.path.join(video_dir, '*.jpg')))
        # if not jpg_paths:
        #     raise RuntimeError('Dataset does not contain .jpg files.')
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
        random_indices = np.random.randint(0, len(jpg_paths), size=(self.K + 1,))
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
        frames = (frames.permute([0, 3, 1, 2]) - 127.5) / 127.5  # K,3,224,224
        marks = (marks.permute([0, 3, 1, 2]) - 127.5) / 127.5  # K,3,224,224
        # frame_mark = frame_mark.requires_grad_(False)

        img = frames[-1]
        mark = marks[-1]
        frames = frames[:self.K]
        marks = marks[:self.K]

        return frames, marks, img, mark, vid_idx


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]
