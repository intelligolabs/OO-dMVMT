import os
import shutil
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from model.OO_dMVMT import dMVMT
from typing import List

from utils.data_loader_shrec22 import Dataset_test_window
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

import sys

sys.path.insert(1, os.getcwd())


# EVAL SCRIPT
def test_our_model(
    model, 
    W,
    output_file_name="output/result.txt", 
    permute=None, 
    dataset_dir='data/shrec22/test_set',
    cfg_path: str='configs/eval_OO-dMVMT.yaml'
):
    args = OmegaConf.load(cfg_path)

    device = torch.device(f"cuda:{args.device_index}" if torch.cuda.is_available() else "cpu")

    normalize = args.normalize
    dataset_step = args.step
    calc_m=args.calc_m
    output_file_content = ''
    for file_index in tqdm(range(1, 145), desc="Generating online eval file"):
        output_file_content += str(file_index) + ";"

        dataset = Dataset_test_window(
            dataset_dir,
            W,
            file_index=file_index,
            step=dataset_step,
            normalize=normalize,
            calc_m=calc_m,            
        )
        
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)

        preds = [16 for _ in range((W // 2))]
        labels = []
        model.eval()
        with torch.no_grad():
            for cnt, d in enumerate(tqdm(dataloader)):
                batch=d['P'].to(device, dtype=torch.float)
                label_sdn = d['Label_SDN'].to(device)
                label = d['Label'].to(device)
                M = d['M']
                if isinstance(M, list):
                    M = None
                else:
                    M = M.to(device)

                if permute:
                    batch = batch.permute(*permute)
                output = model(P=batch, M=M)
                sdn_label, output, (ge_out, gs_out) = output
                output_argmax = output.cpu().detach().numpy().argmax(axis=1)
                preds = np.concatenate([preds, output_argmax], axis=0)

                [labels.append(l.item()) for l in label]

        preds = np.concatenate(
            [preds, np.array([16 for _ in range((W // 2) + W % 2)])], axis=0
        )

        start = 0
        lab = 16

        for window_index in range(len(preds)):

            window = preds[window_index : window_index + W]
            window = list(np.bincount(window))
            most_seen_label = window.index(max(window))

            if most_seen_label != 16:
                if lab == 16:
                    
                    start = window_index + W // 2
                lab = most_seen_label
            if most_seen_label == 16 and lab != 16:
                output_file_content += "%s;%i;%i;%i;" % (
                    dataset.label_map[lab],
                    start,
                    window_index + W // 2,
                    start + W // 2,
                )
                lab = 16
        output_file_content += "\n"

        # updating file while looping, to check in real-time
        with open(output_file_name, "w") as f:
            f.write(output_file_content)

    with open(output_file_name, "w") as f:
        f.write(output_file_content)

# MAIN SCRIPT
def eval_ours(
        tests: List[str] = ['output/train_shrec22_oo-dmvmt'],
        cfg_paths: str=['configs/eval_OO-dMVMT.yaml']
):
    
    
    for exp, cfg_path in zip(tests, cfg_paths):

        args = OmegaConf.load(cfg_path)
        dataset_dir = args.dataset_test_dir

        device = torch.device(f"cuda:{args.device_index}" if torch.cuda.is_available() else "cpu")

        W = args.W
        joints_number = args.joints_number
        joints_channels = args.joints_channels
        embedding_dim = int((joints_number - 1) * joints_number / 2)
        filters_num = args.filters_num
        num_classes = args.num_classes

        chkpt = f"{exp}/best_classifier_model.pth"
        if not os.path.isfile(chkpt):
            print(f"File not found: ", chkpt)
            continue

        frames_in = W

        try:
            model = dMVMT(
                frame_l=W,
                joint_n=joints_number,
                joint_d=joints_channels,
                feat_d=embedding_dim,
                filters=filters_num,
                class_num=num_classes,
            )
            model.load_state_dict(torch.load(chkpt))
            model = model.to(device)
        except Exception as e:
            print("Error loading dict / running test")
            continue
        print("Testing model: ", chkpt)

        result_txt = chkpt.replace("best_classifier_model.pth", "result.txt")

        assert ".txt" in result_txt[-4:]
        if os.path.isfile(result_txt) and os.path.getsize(result_txt) > 0:
            results_timing = os.path.getmtime(result_txt)
            chkpt_timing = os.path.getmtime(chkpt)
            if chkpt_timing > results_timing:
                print("Checkpoint newer? Backuping and continue")
                shutil.copy(result_txt, result_txt.replace(".txt", ".older.txt"))
            else:
                print("alread processed")
                print("--" * 50)
                continue
        test_our_model(
            model,
            frames_in,
            output_file_name=result_txt,
            permute=(0, 1, 3, 2),
            dataset_dir=dataset_dir,
            cfg_path=cfg_path
        )

        del model
        print("--" * 50)

if __name__ == "__main__":
    eval_ours(tests=['output/train_shrec22_oo-dmvmt'], cfg_paths=['configs/eval_OO-dMVMT.yaml'])
