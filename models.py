import math
import time

import geoopt.manifolds.stereographic.math as gmath
import torch
import torch.nn as nn
import torch.nn.functional as F

import losses
from backbone.convrnn import ConvGRU
from backbone.hyrnn_nets import MobiusLinear, MobiusDist2Hyperplane
from backbone.select_backbone import select_resnet


def _initialize_weights(module, gain=1.):
    for name, param in module.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain)
    # other resnet weights have been initialized in resnet itself


class Model(nn.Module):
    '''DPC with RNN'''

    def __init__(self, args):

        self.args = args

        super(Model, self).__init__()

        self.last_duration = int(math.ceil(args.seq_len / 4))
        self.last_size = int(math.ceil(args.img_dim / 32))

        self.target = self.sizes_mask = None  # Only used if cross_gpu_score is True. Otherwise they are in trainer

        self.backbone, self.param = select_resnet(args.network_feature,
                                                  track_running_stats=not args.not_track_running_stats)

        self.feature_dim = self.param['feature_size'] if args.feature_dim == -1 else args.feature_dim
        self.adapt_dim = \
            nn.Linear(self.param['feature_size'], args.feature_dim) if args.feature_dim != -1 else nn.Identity()

        self.num_layers = 1  # param for GRU

        """
        When using a ConvGRU with a 1x1 convolution, it is equivalent to using a regular GRU by flattening the H and W 
        dimensions and adding those as extra samples in the batch (B' = BxHxW), and then going back to the original 
        shape. So we can use the hyperbolic GRU.
        """
        if args.hyperbolic:
            self.hyperbolic_linear = MobiusLinear(self.feature_dim,
                                                  self.feature_dim if not args.final_2dim else 2,
                                                  # This computes an exmap0 after the operation, where the linear
                                                  # operation operates in the Euclidean space.
                                                  hyperbolic_input=False,
                                                  hyperbolic_bias=True,
                                                  nonlin=None,  # For now
                                                  fp64_hyper=args.fp64_hyper
                                                  )
            if args.fp64_hyper:
                self.hyperbolic_linear = self.hyperbolic_linear.double()
        self.fp64_hyper = args.fp64_hyper

        self.agg = ConvGRU(input_size=self.feature_dim,
                           hidden_size=self.feature_dim,
                           kernel_size=1,
                           num_layers=self.num_layers)

        if args.use_labels:
            if args.hyperbolic:
                self.network_class = \
                    MobiusDist2Hyperplane(self.feature_dim if not args.final_2dim else 2, args.n_classes)
            else:
                self.network_class = nn.Linear(self.feature_dim, args.n_classes)
        if not args.use_labels or self.args.linear_input == 'predictions_z_hat':
            self.network_pred = nn.Sequential(
                nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1, padding=0)
            )
            _initialize_weights(self.network_pred)

        # If the task is predicting the last subaction, we need some indexing of how far it is
        if self.args.early_action_self or (self.args.early_action and not self.args.action_level_gt):
            self.time_index = nn.Embedding(self.args.num_seq, self.feature_dim)

        self.mask = None
        self.relu = nn.ReLU(inplace=False)
        _initialize_weights(self.agg)

    def forward(self, block, labels=None):
        # block: [B, N, C, SL, W, H]
        (B, N, C, SL, H, W) = block.shape

        # ----------- STEP 1: compute features ------- #
        # features_dist are the features used to compute the distance
        # features_g are the features used to input to the prediction network

        block = block.view(B * N, C, SL, H, W)
        feature = self.backbone(block)
        feature = self.adapt_dim(feature.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        del block
        feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=(1, 1, 1))
        if self.args.no_spatial:
            feature = feature.mean(dim=[-1, -2], keepdims=True)
            self.last_size = 1

        if self.args.hyperbolic:
            feature_reshape = feature.permute(0, 2, 3, 4, 1)
            mid_feature_shape = feature_reshape.shape
            if self.args.final_2dim:
                mid_feature_shape = mid_feature_shape[:-1] + (2,)
            feature_reshape = feature_reshape.reshape(-1, feature.shape[1])
            feature_dist = self.hyperbolic_linear(feature_reshape)  # performs exmap0
            feature_dist = feature_dist.reshape(mid_feature_shape).permute(0, 4, 1, 2, 3)

            if self.args.hyperbolic_version == 1:
                # Do not modify Euclidean feature for g, but compute hyperbolic version for the distance later
                feature_g = feature

            else:  # hyperbolic version 2
                # Move to hyperbolic with linear layer, then create feature_g from there (back to Euclidean)
                # project back to euclidean
                feature_g = gmath.logmap0(feature_dist, k=torch.tensor(-1.), dim=1).float()
        else:
            feature_dist = feature_g = feature

        # before ReLU, (-inf, +inf)
        feature_dist = feature_dist.view(B, N, 2 if self.args.final_2dim else self.feature_dim,
                                         self.last_size, self.last_size)
        feature_predict_from = feature_dist  # To train linear layer on top of
        # And these are the features we have to "predict to" (in the self-supervised setting)
        feature_dist = feature_dist[:, N - self.args.pred_step::, :].contiguous()
        feature_dist = feature_dist.permute(0, 1, 3, 4, 2).reshape(B * self.args.pred_step * self.last_size ** 2,
                                                                   2 if self.args.final_2dim else self.feature_dim)

        # ----------- STEP 2: compute predictions ------- #

        feature = self.relu(feature_g)  # [0, +inf)
        # [B,N,D,6,6], [0, +inf)
        feature = feature.view(B, N, self.feature_dim, self.last_size, self.last_size)

        # If the task is predicting the last subaction, we need some indexing of how far it is
        if self.args.early_action_self or (self.args.early_action and not self.args.action_level_gt):
            feature += self.time_index(torch.range(0, feature.shape[1] - 1).long().to('cuda'))[None, :, :, None, None]

        hidden_all, hidden = self.agg(feature[:, 0:N - self.args.pred_step, :].contiguous())
        hidden = hidden[:, -1, :]  # after tanh, (-1,1). get the hidden state of last layer, last time step

        if self.args.use_labels:
            if self.args.linear_input == 'features_z':
                input_linear = feature_predict_from.mean(dim=[-2, -1])  # just pool spatially
            elif self.args.linear_input == 'predictions_c':
                input_linear = hidden_all.mean(dim=[-2, -1])  # just pool spatially
            else:  # 'predictions_z_hat'
                # project to "features" space
                hidden_all_projected = self.network_pred(hidden_all.view([-1] + list(hidden.shape[1:]))). \
                    view_as(hidden_all)
                input_linear = hidden_all_projected.mean(dim=[-2, -1])  # just pool spatially
            if self.args.action_level_gt and not self.args.early_action:
                # if we use features_z, this is only using the last one. But an assert in main.py controls for that
                input_linear = input_linear[:, -1]
            else:
                input_linear = input_linear.view(-1, hidden_all.shape[2])  # prepare for linear layer
            # Predict label supervisedly
            if self.args.hyperbolic:
                feature_shape = input_linear.shape
                input_linear = input_linear.view(-1, feature_shape[-1])
                if self.args.fp64_hyper:
                    input_linear = input_linear.double()
                input_linear = self.hyperbolic_linear(input_linear)
                if self.args.final_2dim:
                    feature_shape = feature_shape[:-1] + (2,)
                input_linear = input_linear.view(feature_shape)
            pred_classes = self.network_class(input_linear)
            pred = pred_classes
            size_pred = 1

        else:
            if self.args.early_action_self:
                # only one step but for all hidden_all, not just the last hidden
                pred = self.network_pred(hidden_all.view([-1] + list(hidden.shape[1:]))).view_as(hidden_all)

            else:
                pred = []
                for i in range(self.args.pred_step):
                    # sequentially pred future
                    p_tmp = self.network_pred(hidden)
                    pred.append(p_tmp)
                    _, hidden = self.agg(self.relu(p_tmp).unsqueeze(1), hidden.unsqueeze(0))
                    hidden = hidden[:, -1, :]
                pred = torch.stack(pred, 1)  # B, pred_step, xxx

                del hidden

            size_pred = pred.shape[1]
            pred = pred.permute(0, 1, 3, 4, 2)
            pred = pred.reshape(-1, pred.shape[-1])

            if self.args.hyperbolic:
                pred = self.hyperbolic_linear(pred)

        sizes_pred = self.last_size, self.args.pred_step, size_pred
        sizes_pred = torch.tensor(sizes_pred).to(pred.device).unsqueeze(0)

        if self.args.cross_gpu_score:  # Return all predictions to compute score for all gpus together
            return pred, feature_dist, sizes_pred
        else:  # Compute scores individually for the data in this gpu
            sizes_pred = sizes_pred.float().mean(0).int()
            if self.target is None:
                self.target, self.sizes_mask = losses.compute_mask(self.args, sizes_pred, labels.shape[0])

            loss, *results = losses.compute_loss(self.args, feature_dist, pred, labels, self.target, sizes_pred,
                                                 self.sizes_mask, labels.shape[0])
            return loss, results

    def reset_mask(self):
        self.mask = None
