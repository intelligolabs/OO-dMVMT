import torch
from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm
import os.path as opt
import re
import pickle as pkl
import time
from utils.views import get_CG


class Dataset_shrec22(Dataset):
    def __init__(
        self,
        data_dir: str,
        w: int,
        step: int=1,
        normalize: bool=True,
        calc_m: bool=True,
    ):
        """
        :param data_dir: path to the dataset directory
        :param w: parameter deciding window size W
        :param step: stride for sliding window
        :param normalize: whether to normalize data using train split mean and variance
        :param calc_m: whether to precompute JCD (to speed up training steps)
        """
        self.path_to_data = data_dir
        self.w = w
        self.sequence = []
        self.labels_window = (
            []
        )
        self.label = []
        self.step = step
        self.normalize = normalize
        self.calc_m = calc_m

        self.label_map = [
            "ONE",
            "TWO",
            "THREE",
            "FOUR",
            "OK",
            "MENU",
            "LEFT",
            "RIGHT",
            "CIRCLE",
            "V",
            "CROSS",
            "GRAB",
            "PINCH",
            "DENY",
            "WAVE",
            "KNOB",
            "nongesture",
        ]

        if normalize:
            mean = torch.tensor(
                [0.0360228490, -0.0765615106, 0.4257422931], dtype=torch.float64
            )
            std = torch.tensor(
                [0.1018783228, 0.0665624935, 0.0807055109], dtype=torch.float64
            )
            self.mean = (
                mean.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            )  # B, frames, coords, joints
            self.std = (
                std.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            )  # B, frames, coords, joints

        with open(opt.join(self.path_to_data, "annotations.txt"), "r") as gt:
            # Each annotation row contains: "sequenceNum;Label;GSFrame;GEFrame;...;Label;GSFrame;GEFrame;"
            for line in gt.readlines():
                # compensating for dataset errors
                gt_line = line[:]
                line = line.split(";")
                # get sequence number, corresponds to filename
                file_name = line[0]
                line = line[1:]
                # all sequences need to have same length
                gt_window = (
                    np.zeros(780) + 16
                )
                # for each gesture
                for index in range(0, len(line), 3):
                    # gesture start frame
                    s = int(line[index + 1])
                    # gesture end frame
                    e = int(line[index + 2])
                    # gesture label
                    lab = line[index]

                    # populate corresponding sequence with the label 
                    gt_window[s : e + 1] = [
                        self.label_map.index(lab) for _ in range(s, e + 1)
                    ]

                file_poses = []  # Shape: [len(file),3,26]

                with open(opt.join(self.path_to_data, f"{file_name}.txt"), "r") as fp:
                    # (N,3,10,26)
                    # Each sequence file contains: 'Frame Index(integer); Time_stamp(float); Joint1_x (float);
                    # Joint1_y; Joint1_z;  Joint1_rotx; Joint1_roty; Joint1_rotz; Joint1_rotw; Joint2_x;Joint2_y; Joint2_z; .....'.
                    for line_idx, line in enumerate(fp.readlines()):
                        line = line.split(";")[
                            2:-1
                        ]  # remove index, timestamp and end-of-line
                        line = (
                            np.reshape(line, (26, 3)).transpose().astype(np.float64)
                        )  # Reshape in (3,26)
                        file_poses.append([line[0], line[1], line[2]])

                file_poses = np.array(file_poses).astype(np.float64)
                # generate windows of size W for current gesture
                for poses_index in range(0, file_poses.shape[0] - self.w, self.step):
                    self.sequence.append(
                        file_poses[poses_index : self.w + poses_index, :, :]
                    )

                    # associate the gt labels for each frame of that window
                    label_window = gt_window[poses_index : self.w + poses_index]
                    # count most frequent label in that window
                    label_count = list(np.bincount(label_window.astype("int64")))
                    # assign that label to the window
                    self.label.append(label_count.index(max(label_count)))
                    # also save window with labels, in case is needed
                    self.labels_window.append(label_window)

        self.len_data = len(self.sequence)

        if self.calc_m:
            # cache M (JCD)
            start = time.time()
            print(f"Caching M with {self.w} input frames.")
            self.Ms = [
                get_CG(
                    self._preprocess(current_sequence).permute(0, 2, 1), self.w
                ).float()
                for current_sequence in self.sequence
            ]
            print(f"Done caching in {time.time()-start} seconds")
        else:
            # empty list means it will be generated at runtime by the model (like at inference time)
            self.Ms = [[] for seq in self.sequence]

        print("Inizialization Complete")

    def is_gesture_static(self, label: int):
        """
        Function for SDN classifier, returns True if the gesture is a static one
        """
        if isinstance(label, torch.Tensor):
            label = label.item()
        return label in [0, 1, 2, 3, 4, 5]

    def is_gesture_dynamic(self, label: int):
        """
        Function for SDN classifier, returns True if the gesture is a dynamic one
        """
        return not self.is_gesture_static(label) and label.item() != 16

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # normalize data with train mean and variance 
        if len(x.shape) == 4:  # B, frames, 3, kpts
            return (x - self.mean) / self.std
        return (x - self.mean[0, :, :, :]) / self.std[0, :, :, :]  # frames, 3, kpts

    def __len__(self):
        return self.len_data

    def _preprocess(self, sequence: np.ndarray):
        # prepare data
        # turn to tensor
        sequence_tensor = torch.from_numpy(sequence)
        if len(sequence_tensor.shape) > len(sequence.shape):
            sequence_tensor = sequence_tensor.unsqueeze(0)
        # frames, 3, kpts
        if self.normalize:
            # normalize if necessary
            sequence_tensor = self._normalize(sequence_tensor)
            # frames, 3, kpts
        return sequence_tensor

    def __getitem__(self, item):
        current_sequence = self.sequence[item]
        current_sequence = self._preprocess(current_sequence)
        window_label = self.labels_window[item]
        single_label = self.label[item]

        gesture_start_idx = 0
        gesture_end_idx = self.w - 1
        is_gs_valid = False
        is_ge_valid = False

        lab = window_label[0]
        for idx, i in enumerate(window_label):
            if i != lab and i != 16:
                # GS
                is_gs_valid = True
                gesture_start_idx = idx
            if i != lab and i == 16:
                # GE
                is_ge_valid = True
                gesture_end_idx = idx - 1

            lab = i

        # normalize gesture start and end idx
        gesture_start = gesture_start_idx / (len(window_label) - 1)
        gesture_end = gesture_end_idx / (len(window_label) - 1)

        # window_label = torch.tensor(window_label).long()
        single_label = torch.tensor(single_label).long()

        if self.calc_m:
            M = self.Ms[item]
        else:
            # if M has not been cached, return empty list
            M = []
        label_sdn = label_to_sdn(single_label)
        return dict(P=current_sequence, Label_SDN=label_sdn, Label=single_label, M=M, 
                    Gesture_start=gesture_start, Gesture_end=gesture_end, Is_GS_valid=is_gs_valid, Is_GE_valid=is_ge_valid,
                    )


def label_to_sdn(labels):
    def get_label(l):
        if l in [0, 1, 2, 3, 4, 5]:
            return 0
        elif l in [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
            return 1
        else:
            assert l == 16
            return 2

    if labels.numel() == 1:
        return torch.tensor(get_label(labels)).long()
    label_sdn = torch.tensor([get_label(l) for l in labels]).long()
    return label_sdn


class Dataset_test_window(Dataset):
    """
    IMPORTANT
    Dataset used for evaluation script ONLY
    """
    def __init__(
        self,
        data_dir,
        w,
        step=1,
        normalize: bool = True,
        file_index=None,
        calc_m=True,
        
    ):
        """
        :param data_dir: path to the dataset directory
        :param w: parameter deciding window size W
        :param step: stride for sliding window
        :param normalize: whether to normalize data using train split mean and variance
        :param file_index: index of sequence
        :param calc_m: whether to precompute JCD (to speed up training steps)
        """
        self.labels_window = (
            []
        )
        self.path_to_data = data_dir
        self.w = w
        self.file_index = file_index
        self.sequence = []
        self.label = []
        self.step = step
        self.normalize = normalize
        self.calc_m = calc_m

        self.label_map = [
            "ONE",
            "TWO",
            "THREE",
            "FOUR",
            "OK",
            "MENU",
            "LEFT",
            "RIGHT",
            "CIRCLE",
            "V",
            "CROSS",
            "GRAB",
            "PINCH",
            "DENY",
            "WAVE",
            "KNOB",
            "nongesture",
        ]

        if normalize:
            mean = torch.tensor(
                [0.0360228490, -0.0765615106, 0.4257422931], dtype=torch.float64
            )
            std = torch.tensor(
                [0.1018783228, 0.0665624935, 0.0807055109], dtype=torch.float64
            )
            self.mean = (
                mean.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            )  # B, frames, coords, joints
            self.std = (
                std.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            )  # B, frames, coords, joints

        GT = []

        with open(opt.join(self.path_to_data, "annotations.txt"), "r") as gt:
            # open annotation to generate GT window labeling
            for line in gt.readlines():
                # only consider line containing sequence we are examining
                if line.split(";")[0] == str(self.file_index):
                    gt_line = line[:]
                    line = line.split(";")
                    file_name = line[0]
                    line = line[1:]
                    # all sequences need to have same length
                    gt_window = (
                        np.zeros(780) + 16
                    )
                    for index in range(0, len(line), 3):
                        s = int(line[index + 1])
                        e = int(line[index + 2])
                        lab = line[index]

                        gt_window[s : e + 1] = [
                            self.label_map.index(lab) for _ in range(s, e + 1)
                        ]

        GT = np.array(GT)
        file = str(file_index) + ".txt"
        file_poses = []  # Shape: [len(file),3,26]

        with open(opt.join(self.path_to_data, file), "r") as fd:
            # (N,3,10,26)
            # Each sequence file contains: 'Frame Index(integer); Time_stamp(float); Joint1_x (float);
            # Joint1_y; Joint1_z;  Joint1_rotx; Joint1_roty; Joint1_rotz; Joint1_rotw; Joint2_x;Joint2_y; Joint2_z; .....'.
            for line in fd.readlines():
                line = line.split(";")[2:-1]
                line = (
                    np.reshape(line, (26, 3)).transpose().astype(np.float64)
                )  # Reshape in (3,26)

                file_poses.append([line[0], line[1], line[2]])

        file_poses = np.array(file_poses).astype(np.float64)

        # generate windows of size W, using sliding window with stride=step
        for poses_index in range(0, file_poses.shape[0] - self.w, self.step):
            self.sequence.append(
                file_poses[poses_index : self.w + poses_index, :, :]
            )

            label_window = gt_window[poses_index : self.w + poses_index]
            label_count = list(np.bincount(label_window.astype("int64")))
            self.label.append(label_count.index(max(label_count)))
            self.labels_window.append(label_window)
        self.len_data = len(self.sequence)

        if self.calc_m:
            start = time.time()
            print(f"Caching M with {self.w} input frames.")
            self.Ms = [
                get_CG(
                    self._preprocess(current_sequence).permute(0, 2, 1), self.w
                ).float()
                for current_sequence in self.sequence
            ]
            print(f"Done caching in {time.time()-start} seconds")
        else:
            self.Ms = []

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 4:  # B, frames, 3, kpts
            return (x - self.mean) / self.std
        return (x - self.mean[0, :, :, :]) / self.std[0, :, :, :]  # frames, 3, kpts


    def __len__(self):
        return self.len_data

    def _preprocess(self, sequence: np.ndarray) -> torch.Tensor:
        sequence_tensor = torch.from_numpy(sequence)
        # check for unbatched input
        if len(sequence_tensor.shape) > len(sequence.shape):
            sequence_tensor = sequence_tensor.unsqueeze(0)
        # frames, 3, kpts
        if self.normalize:
            sequence_tensor = self._normalize(sequence_tensor)
            # frames, 6, kpts
        return sequence_tensor

    def __getitem__(self, item):
        current_sequence = self.sequence[item]
        current_sequence = self._preprocess(current_sequence)
        window_label = self.labels_window[item]
        single_label = self.label[item]

        gesture_start_idx = 0
        gesture_end_idx = self.w - 1
        is_gs_valid = False
        is_ge_valid = False

        lab = window_label[0]
        for idx, i in enumerate(window_label):
            if i != lab and i != 16:
                # GS
                is_gs_valid = True
                gesture_start_idx = idx
            if i != lab and i == 16:
                # GE
                is_ge_valid = True
                gesture_end_idx = idx - 1

            lab = i

        gesture_start = gesture_start_idx / (len(window_label) - 1)
        gesture_end = gesture_end_idx / (len(window_label) - 1)

        single_label = torch.tensor(single_label).long()

        if self.calc_m:
            M = self.Ms[item]
        else:
            M = []

        label_sdn = label_to_sdn(single_label)

        return dict(P=current_sequence, Label_SDN=label_sdn, Label=single_label, M=M, 
                    Gesture_start=gesture_start, Gesture_end=gesture_end, Is_GS_valid=is_gs_valid, Is_GE_valid=is_ge_valid,
                    )

if __name__ == "__main__":

    from torch.utils.data import DataLoader
    dataset = Dataset_test_window("data/shrec22/test_set", 16, file_index=10 ,step=1, calc_m=True, 
                                  )
    dataloader = DataLoader(dataset, batch_size=50, num_workers=0)
    last_fore = None

    n_samples = 0
    n_gesture_start = 0
    n_gesture_end = 0

    for d in tqdm(dataloader):
    
        P=d['P']
        label_SDN = d['Label_SDN']
        label = d['Label']
        M = d['M']
        gesture_start = d['Gesture_start']
        gesture_end = d['Gesture_end']
        is_GS_valid = d['Is_GS_valid']
        is_GE_valid= d['Is_GE_valid']
        continue

    print()