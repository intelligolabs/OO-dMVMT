#! /usr/bin/env python
#! coding:utf-8

import time
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import math
from utils.views import get_CG

def poses_diff(x):
    _, H, W, _ = x.shape

    # x.shape (batch,channel,joint_num,joint_dim)
    x = x[:, 1:, ...] - x[:, :-1, ...]
    x = x.permute(0, 3, 1, 2)
    # x.shape (batch,joint_dim,channel,joint_num,)
    x = F.interpolate(x, size=(H, W), align_corners=False, mode="bilinear")
    x = x.permute(0, 2, 3, 1)
    # x.shape (batch,channel,joint_num,joint_dim)
    return x


def poses_motion(P):
    P_diff_slow = poses_diff(P)
    P_diff_slow = torch.flatten(P_diff_slow, start_dim=2)
    P_fast = P[:, ::2, :, :]
    P_diff_fast = poses_diff(P_fast)
    P_diff_fast = torch.flatten(P_diff_fast, start_dim=2)
    return P_diff_slow, P_diff_fast


class c1D(nn.Module):
    # input (B,C,D) //batch,channels,dims
    # output = (B,C,filters)
    def __init__(self, input_channels, input_dims, filters, kernel):
        super(c1D, self).__init__()
        self.cut_last_element = kernel % 2 == 0
        self.padding = math.ceil((kernel - 1) / 2)
        self.conv1 = nn.Conv1d(
            input_dims, filters, kernel, bias=False, padding=self.padding
        )
        self.bn = nn.BatchNorm1d(num_features=input_channels)

    def forward(self, x):
        # x (B,D,C)
        x = x.permute(0, 2, 1)
        # output (B,filters,C)
        if self.cut_last_element:
            output = self.conv1(x)[:, :, :-1]
        else:
            output = self.conv1(x)
        # output = (B,C,filters)
        output = output.permute(0, 2, 1)
        output = self.bn(output)
        output = F.leaky_relu(output, 0.2, True)
        return output


class block(nn.Module):
    def __init__(self, input_channels, input_dims, filters, kernel):
        super(block, self).__init__()
        self.c1D1 = c1D(input_channels, input_dims, filters, kernel)
        self.c1D2 = c1D(input_channels, filters, filters, kernel)

    def forward(self, x):
        output = self.c1D1(x)
        output = self.c1D2(output)
        return output


class d1D(nn.Module):
    def __init__(self, input_dims, filters):
        super(d1D, self).__init__()
        self.linear = nn.Linear(input_dims, filters)
        self.bn = nn.BatchNorm1d(num_features=filters)

    def forward(self, x):
        output = self.linear(x)
        output = self.bn(output)
        output = F.leaky_relu(output, 0.2)
        return output


class spatialDropout1D(nn.Module):
    def __init__(self, p):
        super(spatialDropout1D, self).__init__()
        self.dropout = nn.Dropout2d(p)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        return x


class Permute(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 1)


class dMVMT(nn.Module):
    def __init__(self, frame_l, joint_n, joint_d, feat_d, filters, class_num):
        super(dMVMT, self).__init__()
        # JCD part
        self.jcd_conv1 = nn.Sequential(
            c1D(frame_l, feat_d, 2 * filters, 1), spatialDropout1D(0.1)
        )
        self.jcd_conv2 = nn.Sequential(
            c1D(frame_l, 2 * filters, filters, 3), spatialDropout1D(0.1)
        )
        self.jcd_conv3 = c1D(frame_l, filters, filters, 1)
        self.jcd_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=2), spatialDropout1D(0.1)
        )

        # diff_slow part
        self.slow_conv1 = nn.Sequential(
            c1D(frame_l, joint_n * joint_d, 2 * filters, 1), spatialDropout1D(0.1)
        )
        self.slow_conv2 = nn.Sequential(
            c1D(frame_l, 2 * filters, filters, 3), spatialDropout1D(0.1)
        )
        self.slow_conv3 = c1D(frame_l, filters, filters, 1)
        self.slow_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=2), spatialDropout1D(0.1)
        )

        # fast_part
        self.fast_conv1 = nn.Sequential(
            c1D(frame_l // 2, joint_n * joint_d, 2 * filters, 1), spatialDropout1D(0.1)
        )
        self.fast_conv2 = nn.Sequential(
            c1D(frame_l // 2, 2 * filters, filters, 3), spatialDropout1D(0.1)
        )
        self.fast_conv3 = nn.Sequential(
            c1D(frame_l // 2, filters, filters, 1), spatialDropout1D(0.1)
        )

        # FG classifier branch
        self.ensamble_feature_extractor_classifier = torch.nn.Sequential(
            block(frame_l // 2, 3 * filters, 2 * filters, 3),
            Permute(),
            nn.MaxPool1d(kernel_size=2),
            spatialDropout1D(0.1),
            Permute(),
            block(frame_l // 4, 2 * filters, 4 * filters, 3),
            Permute(),
            nn.MaxPool1d(kernel_size=2),
            spatialDropout1D(0.1),
            Permute(),
            block(frame_l // 8, 4 * filters, 8 * filters, 3),
            spatialDropout1D(0.1),
        )

        # FG classifier head
        self.classifier = torch.nn.Sequential(
            d1D(8 * filters, 128),
            nn.Dropout(0.5),
            d1D(128, 128),
            nn.Dropout(0.5),
            nn.Linear(128, class_num),
        )

        # SDN classifier branch
        self.ensamble_feature_extractor_sdn = torch.nn.Sequential(
            block(frame_l // 2, 3 * filters, 2 * filters, 3),
            Permute(),
            nn.MaxPool1d(kernel_size=2),
            spatialDropout1D(0.1),
            Permute(),
            block(frame_l // 4, 2 * filters, 4 * filters, 3),
            Permute(),
            nn.MaxPool1d(kernel_size=2),
            spatialDropout1D(0.1),
            Permute(),
            block(frame_l // 8, 4 * filters, 8 * filters, 3),
            spatialDropout1D(0.1),
        )

        # SDN classifier head
        self.sdn = torch.nn.Sequential(
            d1D(8 * filters, 128),
            nn.Dropout(0.5),
            d1D(128, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 3),
        )

        # Gesture Start Regressor branch
        self.ensamble_feature_extractor_GS = torch.nn.Sequential(
            block(frame_l // 2, 3 * filters, 2 * filters, 3),
            Permute(),
            nn.MaxPool1d(kernel_size=2),
            spatialDropout1D(0.1),
            Permute(),
            block(frame_l // 4, 2 * filters, 4 * filters, 3),
            Permute(),
            nn.MaxPool1d(kernel_size=2),
            spatialDropout1D(0.1),
            Permute(),
            block(frame_l // 8, 4 * filters, 8 * filters, 3),
            spatialDropout1D(0.1),
        )

        # Gesture Start Regresson head
        self.GS_regr = torch.nn.Sequential(
            d1D(8 * filters, 128),
            nn.Dropout(0.5),
            d1D(128, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

        # Gesture End Regressor branch
        self.ensamble_feature_extractor_GE = torch.nn.Sequential(
            block(frame_l // 2, 3 * filters, 2 * filters, 3),
            Permute(),
            nn.MaxPool1d(kernel_size=2),
            spatialDropout1D(0.1),
            Permute(),
            block(frame_l // 4, 2 * filters, 4 * filters, 3),
            Permute(),
            nn.MaxPool1d(kernel_size=2),
            spatialDropout1D(0.1),
            Permute(),
            block(frame_l // 8, 4 * filters, 8 * filters, 3),
            spatialDropout1D(0.1),
        )

        # Gesture End Regressor Head
        self.GE_regr = torch.nn.Sequential(
            d1D(8 * filters, 128),
            nn.Dropout(0.5),
            d1D(128, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, P: torch.Tensor, M=None):
        if M is None:
            M = torch.cat(
                [get_CG(P[i, :, :, :]).unsqueeze(0) for i in range(P.shape[0])],
                dim=0,
            )
            M = M.to(P.device)

        # extract JCD features
        x = self.jcd_conv1(M)
        x = self.jcd_conv2(x)
        x = self.jcd_conv3(x)
        x = x.permute(0, 2, 1)

        x = self.jcd_pool(x)
        x = x.permute(0, 2, 1)

        # extract fast/slow views
        diff_slow, diff_fast = poses_motion(P)
        x_d_slow = self.slow_conv1(diff_slow)
        x_d_slow = self.slow_conv2(x_d_slow)
        x_d_slow = self.slow_conv3(x_d_slow)
        x_d_slow = x_d_slow.permute(0, 2, 1)  # [B, JC, F]  # J*C filtered
        x_d_slow = self.slow_pool(x_d_slow)
        x_d_slow = x_d_slow.permute(0, 2, 1)  # [B, F, JC]  # J*C filtered

        x_d_fast = self.fast_conv1(diff_fast)
        x_d_fast = self.fast_conv2(x_d_fast)
        x_d_fast = self.fast_conv3(x_d_fast)
        # x_d_fast / x_d_slow -> [B, 8, 8] # [B, F, JC]

        # Ensemble the multi-view features
        ensamble = torch.cat((x, x_d_slow, x_d_fast), dim=2)
        # ensamble.shape : [B, 8, 24]

        # extract features for FG classifier
        ensamble_class = self.ensamble_feature_extractor_classifier(ensamble)
        # max pool over (B,C,D) C channels
        out_class_feat, _idxs_ = torch.max(ensamble_class, dim=1)
        # predict gesture FG classes
        x = self.classifier(out_class_feat)

        # extract features for SDN classifier
        ensamble_sdn = self.ensamble_feature_extractor_sdn(ensamble)
        # max pool over (B,C,D) C channels
        out_sdn_feat, _idxs_ = torch.max(ensamble_sdn, dim=1)
        # predict gesture SDN classes
        sdn = self.sdn(out_sdn_feat)

        # extract features for Gesture Start regressor 
        ensamble_gs_feats = self.ensamble_feature_extractor_GS(ensamble)
        # max pool over (B,C,D) C channels
        gs_feats = torch.max(ensamble_gs_feats, dim=1).values
        # regress gesture start frame
        gs = self.GS_regr(gs_feats)

        # extract features for Gesture End regressor 
        ensamble_ge_feats = self.ensamble_feature_extractor_GE(ensamble)
        # max pool over (B,C,D) C channels
        ge_feats = torch.max(ensamble_ge_feats, dim=1).values
        # regress gesture end frame
        ge = self.GE_regr(ge_feats)
        return sdn, x, (gs, ge)

    def inference(self, P: torch.Tensor = None, M=None):
        if M is None:
            M = torch.cat(
                [get_CG(P[i, :, :, :]).unsqueeze(0) for i in range(P.shape[0])],
                dim=0,
            )
            M = M.to(P.device)

        # extract JCD features
        x = self.jcd_conv1(M)
        x = self.jcd_conv2(x)
        x = self.jcd_conv3(x)
        x = x.permute(0, 2, 1)

        x = self.jcd_pool(x)
        x = x.permute(0, 2, 1)

        # extract fast/slow views
        diff_slow, diff_fast = poses_motion(P)
        x_d_slow = self.slow_conv1(diff_slow)
        x_d_slow = self.slow_conv2(x_d_slow)
        x_d_slow = self.slow_conv3(x_d_slow)
        x_d_slow = x_d_slow.permute(0, 2, 1)  # [B, JC, F]  # J*C filtered
        x_d_slow = self.slow_pool(x_d_slow)
        x_d_slow = x_d_slow.permute(0, 2, 1)  # [B, F, JC]  # J*C filtered

        x_d_fast = self.fast_conv1(diff_fast)
        x_d_fast = self.fast_conv2(x_d_fast)
        x_d_fast = self.fast_conv3(x_d_fast)
        # x_d_fast / x_d_slow -> [B, 8, 8] # [B, F, JC]

        # Ensemble the multi-view features
        ensamble = torch.cat((x, x_d_slow, x_d_fast), dim=2)
        # ensamble.shape : [B, 8, 24]

        # extract features for FG classifier
        ensamble_class = self.ensamble_feature_extractor_classifier(ensamble)
        # max pool over (B,C,D) C channels
        out_class_feat, _idxs_ = torch.max(ensamble_class, dim=1)
        # predict gesture FG classes
        x = self.classifier(out_class_feat)

        return None, x, (None, None)


def __test__():
    frames_in = 16
    joints_number = 26
    joints_channels = 3
    embedding_dim = int((joints_number - 1) * joints_number / 2)
    filters_num = 8
    num_classes = 17

    assert frames_in % 2 == 0

    model: dMVMT = dMVMT(
        frame_l=frames_in,
        joint_n=joints_number,
        joint_d=joints_channels,
        feat_d=embedding_dim,
        filters=filters_num,
        class_num=num_classes,
    )
    batchsize = 1
    x_in_shape = (batchsize, frames_in, joints_number, joints_channels)
    m_in_shape = (batchsize, frames_in, embedding_dim)
    P = torch.randn(*x_in_shape)
    M = torch.randn(*m_in_shape)
    model = model.cuda()
    M = M.cuda()

    model.eval()

    runs = 100
    with torch.no_grad():
        _ = model(P.cuda(), M=M)  # warmup

        timings = []
        for i in range(runs):
            P = torch.randn(*x_in_shape)
            P = P.cuda()
            start = time.time()
            _ = model.inference(P)
            elapsed = time.time() - start
            timings.append(elapsed)

        timings_with_m = []
        for i in range(runs):
            P = torch.randn(*x_in_shape)
            P = P.cuda()
            M = torch.randn(*m_in_shape)
            M = M.cuda()
            start = time.time()
            _ = model.inference(P, M)
            elapsed = time.time() - start
            timings_with_m.append(elapsed)

    print("Timings Realtime", np.mean(timings), "FPS: ", 1 / np.mean(timings))
    print(
        "Timings Realtime - with M",
        np.mean(timings_with_m),
        "FPS: ",
        1 / np.mean(timings_with_m),
    )


if __name__ == "__main__":
    __test__()
