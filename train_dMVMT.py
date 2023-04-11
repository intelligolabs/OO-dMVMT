import random
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from torch.utils.data import DataLoader

import torch.autograd
import torch
import os
from torch import nn
import numpy as np
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf
from utils.results import *
from utils.data_loader_shrec22 import Dataset_shrec22
from model.OO_dMVMT import dMVMT
from fairseq.optim.adafactor import Adafactor
import warnings

warnings.filterwarnings("ignore")




def train(
    model: torch.nn.Module,
    exp: str,
    W: int,
    normalize: bool = True,
    n_epochs: int = 100,
    dataset_step: int = 1,
    do_gs_ge: bool = True,
    do_gs_ge_onOff: bool = True,
    do_sdn: bool = True,
    do_onOff_skip: bool = False,
    save_metrics: bool = True,
    calc_m: bool = True,
    device: torch.device='cpu'
):

    optimizer = Adafactor(model.parameters())
    scheduler = None

    dataset = Dataset_shrec22(
        dataset_train_dir,
        W,
        step=dataset_step,
        normalize=normalize,
        calc_m=calc_m, 
    )
    data_loader = DataLoader(
        dataset,
        batch_size=train_batch_sz,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    valid_dataset = Dataset_shrec22(
        dataset_test_dir,
        W,
        step=dataset_step,
        normalize=normalize,
        calc_m=calc_m, 
    )
    vald_loader = DataLoader(
        valid_dataset,
        batch_size=test_batch_sz,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    print(f">>> EXPERIMENT: {exp}")
    print(">>> Training dataset length: {:d}".format(dataset.__len__()))
    print(">>> Validation dataset length: {:d}".format(valid_dataset.__len__()))

    save_folder_path = os.path.join("output", exp)
    os.makedirs(save_folder_path, exist_ok=True)


    criterion = nn.CrossEntropyLoss()

    best_test_accuracy = 0.0
    best_test_f1 = 0.0
    loss_train = []
    loss_val = []
    val_accuracy = []

    N = 1 + (1 if do_sdn else 0) + (2 if do_gs_ge else 0)  # numero di task

    for epoch in range(n_epochs):

        train_accuracy = []
        test_accuracy = []
        train_all_preds = []
        train_all_gt = []
        test_all_preds = []
        test_all_gt = []
        test_all_sdn_preds = []
        test_all_sdn_gt = []
        train_all_sdn_preds = []
        train_all_sdn_gt = []

        train_all_gs_pred = []
        train_all_gs_gt = []
        train_all_ge_pred = []
        train_all_ge_gt = []

        test_all_gs_pred = []
        test_all_gs_gt = []
        test_all_ge_pred = []
        test_all_ge_gt = []

        running_loss = 0.0
        running_test_loss = 0.0

        model.train()
        for cnt, d in enumerate(
            tqdm(data_loader, desc="Train Epoch " + str(epoch))
        ):
            batch=d['P'].to(device, dtype=torch.float).permute(0, 1, 3, 2)
            label_sdn = d['Label_SDN'].to(device)
            label = d['Label'].to(device)
            M = d['M']
            if isinstance(M, list):
                M = None
            else:
                M = M.to(device)

            gesture_start = d['Gesture_start'].float().to(device)
            gesture_end = d['Gesture_end'].float().to(device)
            is_gs_valid = d['Is_GS_valid'].to(device)
            is_ge_valid= d['Is_GE_valid'].to(device)


            optimizer.zero_grad()

            output = model(batch, M)
            sdn_out, pred, gs_ge = output
            gs, ge = gs_ge

            classification_loss = criterion(pred, label)
            sdn_loss = torch.nn.functional.cross_entropy(sdn_out, label_sdn)

            # use only classifiers, but skip windows that are not gs/ge
            if do_gs_ge and do_onOff_skip:
                pred = pred[is_gs_valid | is_ge_valid]
                label = label[is_gs_valid | is_ge_valid]
                sdn_out = sdn_out[is_gs_valid | is_ge_valid]
                label_sdn = label_sdn[is_gs_valid | is_ge_valid]
                
                if len(pred) == 0:
                    continue

                loss = criterion(pred, label)
                sdn_loss = torch.nn.functional.cross_entropy(sdn_out, label_sdn)

            else:
                loss = classification_loss

                if do_gs_ge:
                    if do_gs_ge_onOff: # default behaviour
                        gs_val: torch.Tensor = gs[is_gs_valid]
                        ge_val: torch.Tensor = ge[is_ge_valid]

                        gesture_start_valid = gesture_start[is_gs_valid]
                        gesture_end_valid = gesture_end[is_ge_valid]
                        if is_gs_valid.any() and gs_val.numel() > 0:
                            gs_loss = torch.nn.functional.mse_loss(gs_val, gesture_start_valid).float()
                            loss = loss + gs_loss

                        if is_ge_valid.any() and ge_val.numel() > 0:
                            ge_loss = torch.nn.functional.mse_loss(ge_val, gesture_end_valid).float()
                            loss = loss + ge_loss

                    else:
                        gs_loss = torch.nn.functional.mse_loss(gs, gesture_start).float()
                        ge_loss = torch.nn.functional.mse_loss(ge, gesture_end).float()
                        loss = loss + gs_loss + ge_loss


            if do_sdn:
                loss = loss + sdn_loss


            loss.backward()
            running_loss += loss.item()

            optimizer.step()
            train_all_preds.append(pred.detach().cpu())
            train_all_gt.append(label.detach().cpu())
            train_all_sdn_preds.append(sdn_out.detach().cpu())
            train_all_sdn_gt.append(label_sdn.detach().cpu())

            train_all_gs_pred.append(gs.detach().cpu())
            train_all_gs_gt.append(gesture_start.detach().cpu())
            train_all_ge_pred.append(ge.detach().cpu())
            train_all_ge_gt.append(gesture_end.detach().cpu())

        loss_train.append(running_loss / (cnt + 1))

        model.eval()
        with torch.no_grad():
            for iii, d in enumerate(
                tqdm(vald_loader, desc="Test Epoch " + str(epoch))
            ):
                batch=d['P'].to(device, dtype=torch.float).permute(0, 1, 3, 2)
                label_sdn = d['Label_SDN'].to(device)
                label = d['Label'].to(device)
                M = d['M']
                if isinstance(M, list):
                    M = None
                else:
                    M = M.to(device)

                gesture_start = d['Gesture_start'].float().to(device)
                gesture_end = d['Gesture_end'].float().to(device)
                is_gs_valid = d['Is_GS_valid'].to(device)
                is_ge_valid= d['Is_GE_valid'].to(device)



                output = model(batch, M)
                sdn_out, pred, gs_ge = output
                gs, ge = gs_ge

                classification_loss = criterion(pred, label)
                sdn_loss = torch.nn.functional.cross_entropy(sdn_out, label_sdn)
                loss = classification_loss

                if do_gs_ge:
                    if do_gs_ge_onOff:
                        gs_val: torch.Tensor = gs[is_gs_valid]
                        ge_val: torch.Tensor = ge[is_ge_valid]

                        gesture_start_valid = gesture_start[is_gs_valid]
                        gesture_end_valid = gesture_end[is_ge_valid]
                        if is_gs_valid.any() and gs_val.numel() > 0:
                            gs_loss = torch.nn.functional.mse_loss(
                                gs_val, gesture_start_valid
                            ).float()
                            loss = loss + gs_loss

                        if is_ge_valid.any() and ge_val.numel() > 0:
                            ge_loss = torch.nn.functional.mse_loss(
                                ge_val, gesture_end_valid
                            ).float()
                            loss = loss + ge_loss

                    else:
                        gs_loss = torch.nn.functional.mse_loss(
                            gs, gesture_start
                        ).float()
                        ge_loss = torch.nn.functional.mse_loss(ge, gesture_end).float()
                        loss = loss + gs_loss + ge_loss

                if do_sdn:
                    loss = loss + sdn_loss

                running_test_loss += loss.item()

                test_all_preds.append(pred.detach().cpu())
                test_all_gt.append(label.detach().cpu())
                test_all_sdn_preds.append(sdn_out.detach().cpu())
                test_all_sdn_gt.append(label_sdn.detach().cpu())

                test_all_gs_pred.append(gs.detach().cpu())
                test_all_gs_gt.append(gesture_start.detach().cpu())
                test_all_ge_pred.append(ge.detach().cpu())
                test_all_ge_gt.append(gesture_end.detach().cpu())

        loss_val.append(running_test_loss / (iii +1))

        train_all_preds = torch.cat(train_all_preds, dim=0).argmax(1).numpy()
        train_all_gt = torch.cat(train_all_gt, dim=0).numpy()
        test_all_preds = torch.cat(test_all_preds, dim=0).argmax(1).numpy()
        test_all_gt = torch.cat(test_all_gt, dim=0).numpy()

        train_all_sdn_preds = torch.cat(train_all_sdn_preds, dim=0).argmax(1).numpy()
        train_all_sdn_gt = torch.cat(train_all_sdn_gt, dim=0).numpy()
        test_all_sdn_preds = torch.cat(test_all_sdn_preds, dim=0).argmax(1).numpy()
        test_all_sdn_gt = torch.cat(test_all_sdn_gt, dim=0).numpy()

        train_all_gs_pred = torch.cat(train_all_gs_pred, dim=0)
        train_all_gs_gt = torch.cat(train_all_gs_gt, dim=0)
        train_all_ge_pred = torch.cat(train_all_ge_pred, dim=0)
        train_all_ge_gt = torch.cat(train_all_ge_gt, dim=0)
        train_gs_error = torch.nn.functional.mse_loss(
            train_all_gs_pred, train_all_gs_gt
        ).item()
        train_ge_error = torch.nn.functional.mse_loss(
            train_all_ge_pred, train_all_ge_gt
        ).item()

        test_all_gs_pred = torch.cat(test_all_gs_pred)
        test_all_gs_gt = torch.cat(test_all_gs_gt)
        test_all_ge_pred = torch.cat(test_all_ge_pred)
        test_all_ge_gt = torch.cat(test_all_ge_gt)
        test_gs_error = torch.nn.functional.mse_loss(
            test_all_gs_pred, test_all_gs_gt
        ).item()
        test_ge_error = torch.nn.functional.mse_loss(
            test_all_ge_pred, test_all_ge_gt
        ).item()

        train_accuracy = accuracy_score(train_all_gt, train_all_preds)
        test_accuracy = accuracy_score(test_all_gt, test_all_preds)
        test_precision = precision_score(
            test_all_gt, test_all_preds, average="weighted"
        )
        test_recall = recall_score(test_all_gt, test_all_preds, average="weighted")
        test_f1 = f1_score(test_all_gt, test_all_preds, average="weighted")

        train_sdn_accuracy = accuracy_score(train_all_sdn_gt, train_all_sdn_preds)
        test_sdn_accuracy = accuracy_score(test_all_sdn_gt, test_all_sdn_preds)
        val_accuracy.append(test_accuracy)

        print("Train accuracy:       ", train_accuracy)
        print("Validation accuracy:  ", test_accuracy)
        print("Validation F1:  ", test_f1)
        print("")
        print("Train SDN accuracy: ", train_sdn_accuracy)
        print("Test SDN accuracy: ", test_sdn_accuracy)

        print("Train Gesture Start error: ", train_gs_error)
        print("Train Gesture End error: ", train_ge_error)

        print("Test Gesture Start error: ", test_gs_error)
        print("Test Gesture End error: ", test_ge_error)

        print("LR: ", optimizer.param_groups[0]["lr"])
        print("Loss: ", running_loss / (cnt + 1))
        print("")
        if scheduler is not None:
            scheduler.step()

        if save_metrics:
            np.save(os.path.join(save_folder_path, "loss_train"), np.asarray(loss_train))
            np.save(os.path.join(save_folder_path, "loss_test"), np.asarray(loss_val))
            np.save(os.path.join(save_folder_path, "val_accuracy"), np.asarray(val_accuracy))
        

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            print("-- Saving best model --\n")
            save_folder_path = os.path.join("output", exp)
            os.makedirs(save_folder_path, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(save_folder_path, "best_classifier_model.pth"),
            )
            train_cm = print_confusion_matrix(
                train_all_gt,
                train_all_preds,
                os.path.join(save_folder_path, "train_ConfMatrix.jpg"),
            )
            test_cm = print_confusion_matrix(
                test_all_gt,
                test_all_preds,
                os.path.join(save_folder_path, "test_ConfMatrix.jpg"),
            )
            if save_metrics:
                np.save(os.path.join(save_folder_path, "class_test_ConfMatrix"), test_cm)
                np.save(os.path.join(save_folder_path, "class_train_ConfMatrix"), train_cm)
            with open(
                os.path.join(save_folder_path, "classification_results.txt"), "w"
            ) as fd:
                fd.write(f"Epoch: {epoch}\n")
                fd.write(f"Train Accuracy: {train_accuracy}\n")
                fd.write(f"Test Accuracy: {test_accuracy}\n")
                fd.write(f"Test Precision: {test_precision}\n")
                fd.write(f"Test Recall: {test_recall}\n")
                fd.write(f"Test F1: {test_f1}\n")
                fd.write(f"Train SDN accuracy: {train_sdn_accuracy}\n")
                fd.write(f"Test SDN accuracy: {test_sdn_accuracy}\n")
                fd.write(f"Train GestureStart error: {train_gs_error}\n")
                fd.write(f"Train GestureEnd error: {train_ge_error}\n")
                fd.write(f"Test GestureStart error: {test_gs_error}\n")
                fd.write(f"Test GestureEnd error: {test_ge_error}\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train OO-dMVMT')
    parser.add_argument('--cfg_path', default='configs/train_OO-dMVMT.yaml', help='Path to the train.yaml config')
    args = parser.parse_args()
    # SETTINGS
    cfg_path = args.cfg_path
    args = OmegaConf.load(cfg_path)
    dataset_train_dir = args.dataset_train_dir
    dataset_test_dir = args.dataset_test_dir

    device = torch.device(f"cuda:{args.device_index}" if torch.cuda.is_available() else "cpu")

    W = args.W
    joints_number = args.joints_number
    joints_channels = args.joints_channels
    embedding_dim = int((joints_number - 1) * joints_number / 2)
    filters_num = args.filters_num
    num_classes = args.num_classes

    test_batch_sz = args.test_batch_sz
    train_batch_sz = args.train_batch_sz

    assert W % 2 == 0
    exp = f"OO-dMVMT_{W:=}" if args.experiment_name is None else args.experiment_name

    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("Using device: %s" % device)

    model: dMVMT = dMVMT(
        frame_l=W,
        joint_n=joints_number,
        joint_d=joints_channels,
        feat_d=embedding_dim,
        filters=filters_num,
        class_num=num_classes,
    ).to(device)

    print(
        "total number of parameters of the network is: "
        + str(sum(p.numel() for p in model.parameters() if p.requires_grad))
    )

    train(
        model = model,
        exp = exp,
        W = W,
        normalize = args.normalize,
        n_epochs = args.n_epochs,
        dataset_step = args.step,
        do_gs_ge = args.do_gs_ge,
        do_gs_ge_onOff = args.do_gs_ge_OnOff,
        do_sdn = args.do_sdn,
        # weighted = False,
        do_onOff_skip = args.do_OnOff_skip,
        save_metrics = args.save_metrics,
        calc_m=args.calc_m,
        device=device
    )
