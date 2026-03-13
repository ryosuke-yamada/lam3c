"""
LAM3C pretraining model.

This module implements the LAM3C pretraining objective on top of the Pointcept
training framework. It extends a Sonata-style teacher-student masked pretraining
pipeline with local smoothness regularization and optional consistency
regularization.
"""

import copy
from functools import partial
from itertools import chain
from typing import Dict, List, Optional, Tuple

from packaging import version
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from timm.layers import trunc_normal_

import pointops
from pointcept.models.builder import MODELS, build_model
from pointcept.models.modules import PointModel
from pointcept.models.utils import offset2batch, offset2bincount, batch2offset
from pointcept.models.utils.structure import Point
from pointcept.utils.comm import get_world_size, all_gather
from pointcept.utils.scheduler import CosineScheduler


class OnlineCluster(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=4096,
        embed_channels=512,
        num_prototypes=4096,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, embed_channels),
        )
        self.apply(self._init_weights)
        if version.parse(torch.__version__) >= version.parse("2.1.0"):
            self.prototype = torch.nn.utils.parametrizations.weight_norm(
                nn.Linear(embed_channels, num_prototypes, bias=False)
            )
            self.prototype.parametrizations.weight.original0.data.fill_(1)
            self.prototype.parametrizations.weight.original0.requires_grad = False

        else:
            self.prototype = torch.nn.utils.weight_norm(
                nn.Linear(embed_channels, num_prototypes, bias=False)
            )
            self.prototype.weight_g.data.fill_(1)
            self.prototype.weight_g.requires_grad = False

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, feat):
        feat = self.mlp(feat)
        eps = 1e-6 if feat.dtype == torch.float16 else 1e-12
        feat = nn.functional.normalize(feat, dim=-1, p=2, eps=eps)
        similarity = self.prototype(feat)
        return similarity


@MODELS.register_module("LAM3C")
class LAM3C(PointModel):
    def __init__(
        self,
        backbone,
        head_in_channels,
        head_hidden_channels=4096,
        head_embed_channels=512,
        head_num_prototypes=4096,
        teacher_custom=None,
        num_global_view=2,
        num_local_view=4,
        mask_size_start=0.1,
        mask_size_base=0.4,
        mask_size_warmup_ratio=0.05,
        mask_ratio_start=0.3,
        mask_ratio_base=0.7,
        mask_ratio_warmup_ratio=0.05,
        mask_jitter=None,
        teacher_temp_start=0.04,
        teacher_temp_base=0.07,
        teacher_temp_warmup_ratio=0.05,
        student_temp=0.1,
        mask_loss_weight=2 / 8,
        roll_mask_loss_weight=2 / 8,
        unmask_loss_weight=4 / 8,
        momentum_base=0.996,
        momentum_final=1,
        match_max_k=8,
        match_max_r=0.08,
        up_cast_level=2,
        # Local smoothness regularization parameters
        laplacian_loss_weight_start=1e-4,
        laplacian_loss_weight_base=5e-3,
        laplacian_loss_weight_warmup_ratio=0.1,
        laplacian_knn=12,
        laplacian_sigma=None,  # Auto-compute from median kNN distance if None
        laplacian_use_gaussian=True,
        laplacian_max_distance=None,  # Cut edges longer than this (optional)
        laplacian_use_huber=False,
        laplacian_huber_delta=1.0,
        # Optional teacher-student consistency loss
        consistency_loss_weight=0.0,
        consistency_augment_strength=0.02,
    ):
        """Initialize LAM3C pretraining model.

        Args:
            teacher_custom: Optional backbone overrides applied to the teacher
                backbone only. Student backbone uses the original `backbone`.
        """
        super().__init__()

        # loss weights
        self.mask_loss_weight = mask_loss_weight
        self.roll_mask_loss_weight = roll_mask_loss_weight
        self.unmask_loss_weight = unmask_loss_weight

        # view settings
        self.num_global_view = num_global_view
        self.num_local_view = num_local_view

        # mask schedulers
        self.mask_size = mask_size_start
        self.mask_size_start = mask_size_start
        self.mask_size_base = mask_size_base
        self.mask_size_warmup_ratio = mask_size_warmup_ratio
        self.mask_size_scheduler = None

        self.mask_ratio = mask_ratio_start
        self.mask_ratio_start = mask_ratio_start
        self.mask_ratio_base = mask_ratio_base
        self.mask_ratio_warmup_ratio = mask_ratio_warmup_ratio
        self.mask_ratio_scheduler = None

        self.mask_jitter = mask_jitter

        # temperature / momentum schedulers
        self.teacher_temp = teacher_temp_start
        self.teacher_temp_start = teacher_temp_start
        self.teacher_temp_base = teacher_temp_base
        self.teacher_temp_warmup_ratio = teacher_temp_warmup_ratio
        self.teacher_temp_scheduler = None
        self.student_temp = student_temp

        # momentum and scheduler
        self.momentum = momentum_base
        self.momentum_base = momentum_base
        self.momentum_final = momentum_final
        self.momentum_scheduler = None

        # matching / feature settings
        self.match_max_k = match_max_k
        self.match_max_r = match_max_r
        self.up_cast_level = up_cast_level

        # local smoothness regularization
        self.laplacian_loss_weight = laplacian_loss_weight_start
        self.laplacian_loss_weight_start = laplacian_loss_weight_start
        self.laplacian_loss_weight_base = laplacian_loss_weight_base
        self.laplacian_loss_weight_warmup_ratio = laplacian_loss_weight_warmup_ratio
        self.laplacian_loss_weight_scheduler = None
        self.laplacian_knn = laplacian_knn
        self.laplacian_sigma = laplacian_sigma
        self.laplacian_use_gaussian = laplacian_use_gaussian
        self.laplacian_max_distance = laplacian_max_distance
        self.laplacian_use_huber = laplacian_use_huber
        self.laplacian_huber_delta = laplacian_huber_delta

        # consistency regularization
        self.consistency_loss_weight = consistency_loss_weight
        self.consistency_augment_strength = consistency_augment_strength

        assert (
            unmask_loss_weight + mask_loss_weight + roll_mask_loss_weight > 0
        ), "At least one of unmask/mask/roll_mask loss weights must be > 0."
        assert (
            num_global_view > 1 or roll_mask_loss_weight == 0
        ), "roll_mask_loss requires num_global_view > 1."
        assert (
            num_global_view in (1, 2)
        ), "Current implementation supports num_global_view in {1, 2}."

        # student / teacher model construction
        student_model_dict = dict()
        teacher_model_dict = dict()
        if teacher_custom is None:
            teacher_custom = {}
        student_backbone_cfg = copy.deepcopy(backbone)
        teacher_backbone_cfg = copy.deepcopy(backbone)
        teacher_backbone_cfg.update(teacher_custom)
        student_backbone = build_model(student_backbone_cfg)
        teacher_backbone = build_model(teacher_backbone_cfg)
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone

        head = partial(
            OnlineCluster,
            in_channels=head_in_channels,
            hidden_channels=head_hidden_channels,
            embed_channels=head_embed_channels,
            num_prototypes=head_num_prototypes,
        )
        if self.mask_loss_weight > 0 or self.roll_mask_loss_weight > 0:
            student_model_dict["mask_head"] = head()
            teacher_model_dict["mask_head"] = head()
        if self.unmask_loss_weight > 0:
            student_model_dict["unmask_head"] = head()
            teacher_model_dict["unmask_head"] = head()

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)
        for k, v in self.student.items():
            self.teacher[k].load_state_dict(self.student[k].state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False

    def before_train(self):
        # make ModelHook after CheckPointLoader
        total_steps = self.trainer.cfg.scheduler.total_steps
        curr_step = self.trainer.start_epoch * len(self.trainer.train_loader)
        # mask size scheduler
        self.mask_size_scheduler = CosineScheduler(
            start_value=self.mask_size_start,
            base_value=self.mask_size_base,
            final_value=self.mask_size_base,
            warmup_iters=int(total_steps * self.mask_size_warmup_ratio),
            total_iters=total_steps,
        )
        self.mask_size_scheduler.iter = curr_step

        # mask ratio scheduler
        self.mask_ratio_scheduler = CosineScheduler(
            start_value=self.mask_ratio_start,
            base_value=self.mask_ratio_base,
            final_value=self.mask_ratio_base,
            warmup_iters=int(total_steps * self.mask_ratio_warmup_ratio),
            total_iters=total_steps,
        )
        self.mask_ratio_scheduler.iter = curr_step

        # teacher temperature scheduler
        self.teacher_temp_scheduler = CosineScheduler(
            start_value=self.teacher_temp_start,
            base_value=self.teacher_temp_base,
            final_value=self.teacher_temp_base,
            warmup_iters=int(total_steps * self.teacher_temp_warmup_ratio),
            total_iters=total_steps,
        )
        self.teacher_temp_scheduler.iter = curr_step

        # momentum scheduler
        self.momentum_scheduler = CosineScheduler(
            base_value=self.momentum_base,
            final_value=self.momentum_final,
            total_iters=total_steps,
        )
        self.momentum_scheduler.iter = curr_step

        # laplacian loss weight scheduler
        if self.laplacian_loss_weight_base > 0:
            self.laplacian_loss_weight_scheduler = CosineScheduler(
                start_value=self.laplacian_loss_weight_start,
                base_value=self.laplacian_loss_weight_base,
                final_value=self.laplacian_loss_weight_base * 0.5,  # Anneal down slightly
                warmup_iters=int(total_steps * self.laplacian_loss_weight_warmup_ratio),
                total_iters=total_steps,
            )
            self.laplacian_loss_weight_scheduler.iter = curr_step

    def before_step(self):
        # update parameters from schedulers
        self.mask_size = self.mask_size_scheduler.step()
        self.mask_ratio = self.mask_ratio_scheduler.step()
        self.teacher_temp = self.teacher_temp_scheduler.step()
        self.momentum = self.momentum_scheduler.step()
        
        if self.laplacian_loss_weight_scheduler is not None:
            self.laplacian_loss_weight = self.laplacian_loss_weight_scheduler.step()

        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar(
                "params/mask_size",
                self.mask_size,
                self.mask_size_scheduler.iter,
            )
            self.trainer.writer.add_scalar(
                "params/mask_ratio",
                self.mask_ratio,
                self.mask_ratio_scheduler.iter,
            )
            self.trainer.writer.add_scalar(
                "params/teacher_temp",
                self.teacher_temp,
                self.teacher_temp_scheduler.iter,
            )
            self.trainer.writer.add_scalar(
                "params/momentum",
                self.momentum,
                self.momentum_scheduler.iter,
            )
            if self.laplacian_loss_weight_scheduler is not None:
                self.trainer.writer.add_scalar(
                    "params/laplacian_loss_weight",
                    self.laplacian_loss_weight,
                    self.laplacian_loss_weight_scheduler.iter,
                )

    def after_step(self):
        # EMA update teacher
        with torch.no_grad():
            m = self.momentum
            student_param_list = list(self.student.parameters())
            teacher_param_list = list(self.teacher.parameters())
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    @staticmethod
    def sinkhorn_knopp(feat, temp, num_iter=3):
        feat = feat.float()
        q = torch.exp(feat / temp).t()
        n = sum(all_gather(q.shape[1]))  # number of samples to assign
        k = q.shape[0]  # number of prototypes

        # make the matrix sums to 1
        sum_q = q.sum()
        if get_world_size() > 1:
            dist.all_reduce(sum_q)
        q = q / sum_q

        for i in range(num_iter):
            # normalize each row: total weight per prototype must be 1/k
            q_row_sum = q.sum(dim=1, keepdim=True)
            if get_world_size() > 1:
                dist.all_reduce(q_row_sum)
            q = q / q_row_sum / k

            # normalize each column: total weight per sample must be 1/n
            q = q / q.sum(dim=0, keepdim=True) / n

        q *= n  # the columns must sum to 1 so that Q is an assignment
        return q.t()

    def generate_mask(
        self, coord: torch.Tensor, offset: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = offset2batch(offset)
        mask_size = self.mask_size
        mask_ratio = self.mask_ratio

        # Grouping points with grid patch
        min_coord = torch_scatter.segment_coo(coord, batch, reduce="min")
        grid_coord = ((coord - min_coord[batch]) // mask_size).int()
        grid_coord = torch.cat([batch.unsqueeze(-1), grid_coord], dim=-1)
        unique, point_cluster, _ = torch.unique(
            grid_coord, dim=0, sorted=True, return_inverse=True, return_counts=True
        )
        patch_num = unique.shape[0]
        mask_patch_num = int(patch_num * mask_ratio)
        patch_index = torch.randperm(patch_num, device=coord.device)
        mask_patch_index = patch_index[:mask_patch_num]
        point_mask = torch.isin(point_cluster, mask_patch_index)
        return point_mask, point_cluster

    @torch.no_grad()
    def match_neighbour(
        self,
        view1_coord: torch.Tensor,
        view1_offset: torch.Tensor,
        view2_coord: torch.Tensor,
        view2_offset: torch.Tensor,
    ) -> torch.Tensor:
        # pointops.knn_query returns SQUARED distances
        index2, distance_sq = pointops.knn_query(
            1,
            view2_coord.float(),
            view2_offset.int(),
            view1_coord.float(),
            view1_offset.int(),
        )
        index1 = torch.arange(
            index2.shape[0], device=index2.device, dtype=torch.long
        ).unsqueeze(-1)
        # Compare with squared threshold to match distance_sq
        threshold_sq = self.match_max_r * self.match_max_r
        index = torch.cat([index1, index2], dim=-1)[
            distance_sq.squeeze(-1) < threshold_sq
        ]
        return index

    @torch.no_grad()
    def roll_point(self, point: Point) -> Point:
        n = self.num_global_view
        # [pc1, pc1', pc2, pc2'] -> [pc1', pc1, pc2', pc2], only support num_global_view == 2
        bs = len(point.offset) // self.num_global_view
        data_dict = {}
        for key in point.keys():
            if key in ["feat", "coord", "origin_coord", "batch"]:
                value = point[key].split(offset2bincount(point.offset).tolist())
                value = chain(*[value[n * b : n * (b + 1)][::-1] for b in range(bs)])
                if key == "batch":
                    value = [torch.ones_like(v) * i for i, v in enumerate(value)]
                data_dict[key] = torch.cat(list(value), dim=0)
        return Point(data_dict)

    def up_cast(self, point: Point) -> Point:
        for _ in range(self.up_cast_level):
            assert "pooling_parent" in point.keys(), "Missing key: pooling_parent"
            assert "pooling_inverse" in point.keys(), "Missing key: pooling_inverse"
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        return point

    def compute_laplacian_smoothness_loss(
        self, embeddings: torch.Tensor, coords: torch.Tensor, offset: torch.Tensor
    ) -> torch.Tensor:
        """Compute degree-normalized graph smoothness regularization."""
        # Cast to float32 for numerical stability (avoid underflow with fp16)
        embeddings = embeddings.float()
        coords = coords.float()
        batch = offset2batch(offset)
        
        # Build k-NN graph using pointops
        # Note: pointops.knn_query returns SQUARED distances
        knn_indices, knn_distances_sq = pointops.knn_query(
            self.laplacian_knn,
            coords,
            offset.int(),
            coords,
            offset.int(),
        )  # knn_indices: (N, k), knn_distances_sq: (N, k) - SQUARED distances
        
        # Compute sigma from median kNN distance if not specified
        if self.laplacian_sigma is None:
            with torch.no_grad():
                # Use median of k-th nearest neighbor distance (take sqrt of squared distance)
                median_dist = torch.sqrt(torch.median(knn_distances_sq[:, -1]) + 1e-8)
                sigma = median_dist.item()
        else:
            sigma = self.laplacian_sigma
        
        # Compute edge weights
        if self.laplacian_use_gaussian:
            # Gaussian weights: w_ij = exp(-||p_i - p_j||^2 / sigma^2)
            # knn_distances_sq is already squared, so use it directly
            weights = torch.exp(-knn_distances_sq / (sigma ** 2 + 1e-8))
        else:
            # Binary weights (0/1)
            weights = torch.ones_like(knn_distances_sq)
        
        # Optional: mask out long-distance edges
        if self.laplacian_max_distance is not None:
            # Compare with squared distance threshold
            edge_mask = knn_distances_sq < (self.laplacian_max_distance ** 2)
            weights = weights * edge_mask.float()
        
        # Compute embedding differences
        N, k = knn_indices.shape
        
        # Remove self-edges (first neighbor is typically the point itself)
        # Create a mask to exclude self-edges where distance is nearly zero
        self_edge_mask = knn_distances_sq > 1e-10  # (N, k)
        weights = weights * self_edge_mask.float()  # Zero out self-edges
        
        # Expand embeddings for neighbors
        neighbor_embeddings = embeddings[knn_indices.view(-1)].view(N, k, -1)  # (N, k, D)
        center_embeddings = embeddings.unsqueeze(1).expand(-1, k, -1)  # (N, k, D)
        
        # Compute squared L2 distance between embeddings
        embed_diff = center_embeddings - neighbor_embeddings  # (N, k, D)
        
        if self.laplacian_use_huber:
            # Huber loss for robustness
            embed_diff_norm = torch.norm(embed_diff, dim=-1)  # (N, k)
            huber_loss = torch.where(
                embed_diff_norm <= self.laplacian_huber_delta,
                0.5 * embed_diff_norm ** 2,
                self.laplacian_huber_delta * (embed_diff_norm - 0.5 * self.laplacian_huber_delta)
            )
            per_edge_loss = huber_loss  # (N, k)
        else:
            # Standard L2 loss
            per_edge_loss = torch.sum(embed_diff ** 2, dim=-1)  # (N, k)
        
        # Degree normalization: normalize by sum of weights (normalized Laplacian)
        # This prevents dense regions from dominating the loss
        # Handle degree-0 points (all edges invalid) safely
        degree = weights.sum(dim=-1)  # (N,) - sum of edge weights
        valid_mask = degree > 0  # Points with at least one valid neighbor
        
        # Initialize point loss as zeros
        point_loss = torch.zeros_like(degree)
        
        # Only compute loss for points with valid neighbors
        if valid_mask.any():
            numerator = (weights[valid_mask] * per_edge_loss[valid_mask]).sum(dim=-1)
            point_loss[valid_mask] = numerator / degree[valid_mask]
        
        # Average per batch, then average over batches
        # Only average over points with valid neighbors
        if valid_mask.any():
            laplacian_loss = torch_scatter.segment_coo(
                point_loss[valid_mask],
                index=batch[valid_mask],
                reduce="mean",
            ).mean()
        else:
            # All points have degree 0 - return zero loss
            laplacian_loss = torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)
        
        return laplacian_loss

    def compute_consistency_loss(self, point_a: Point, point_b: Point) -> torch.Tensor:
        """Compute teacher-student consistency loss between augmented views."""
        # Match nearest neighbors between two views
        match_index = self.match_neighbour(
            point_b.origin_coord,
            point_b.offset,
            point_a.origin_coord,
            point_a.offset,
        )
        
        # Compute L2 distance between matched embeddings
        teacher_embed = point_a.feat[match_index[:, 1]]
        student_embed = point_b.feat[match_index[:, 0]]
        
        # Normalize embeddings
        eps = 1e-6 if teacher_embed.dtype == torch.float16 else 1e-12
        teacher_embed = F.normalize(teacher_embed, dim=-1, p=2, eps=eps)
        student_embed = F.normalize(student_embed, dim=-1, p=2, eps=eps)
        
        # MSE loss
        consistency_loss = F.mse_loss(student_embed, teacher_embed, reduction="none")
        consistency_loss = consistency_loss.sum(dim=-1)  # Sum over feature dimension
        
        # Average per batch
        consistency_loss = torch_scatter.segment_coo(
            consistency_loss,
            index=point_b.batch[match_index[:, 0]],
            reduce="mean",
        ).mean()
        
        return consistency_loss

    def _prepare_points(
        self, data_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Point, Point, Point]:
        global_point = Point(
            feat=data_dict["global_feat"],
            coord=data_dict["global_coord"],
            origin_coord=data_dict["global_origin_coord"],
            offset=data_dict["global_offset"],
            grid_size=data_dict["grid_size"][0],
        )
        global_mask, _ = self.generate_mask(global_point.coord, global_point.offset)
        mask_global_coord = global_point.coord.clone().detach()
        if self.mask_jitter is not None:
            mask_global_coord[global_mask] += torch.clip(
                torch.randn_like(mask_global_coord[global_mask]).mul(self.mask_jitter),
                max=self.mask_jitter * 2,
            )
        mask_global_point = Point(
            feat=data_dict["global_feat"],
            coord=mask_global_coord,
            origin_coord=data_dict["global_origin_coord"],
            mask=global_mask,
            offset=data_dict["global_offset"],
            grid_size=data_dict["grid_size"][0],
        )
        local_point = Point(
            feat=data_dict["local_feat"],
            coord=data_dict["local_coord"],
            origin_coord=data_dict["local_origin_coord"],
            offset=data_dict["local_offset"],
            grid_size=data_dict["grid_size"][0],
        )
        return global_point, mask_global_point, local_point

    def _compute_mask_losses(
        self,
        global_point_teacher: Point,
        global_feat: torch.Tensor,
        mask_global_point: Point,
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor], Optional[Point]]:
        metrics: Dict[str, torch.Tensor] = {}
        loss_terms: List[torch.Tensor] = []
        if self.mask_loss_weight <= 0 and self.roll_mask_loss_weight <= 0:
            return metrics, loss_terms, None

        with torch.no_grad():
            global_point_teacher.feat = self.teacher.mask_head(global_feat)
        mask_global_point_student = self.student.backbone(mask_global_point)
        mask_global_point_student = self.up_cast(mask_global_point_student)
        mask_pred_sim = self.student.mask_head(mask_global_point_student.feat)

        if self.mask_loss_weight > 0:
            with torch.no_grad():
                match_index = self.match_neighbour(
                    mask_global_point_student.origin_coord,
                    mask_global_point_student.offset,
                    global_point_teacher.origin_coord,
                    global_point_teacher.offset,
                )
                mask_target_sim = self.sinkhorn_knopp(
                    global_point_teacher.feat[match_index[:, 1]],
                    self.teacher_temp,
                )
            mask_loss = -torch.sum(
                mask_target_sim
                * F.log_softmax(
                    mask_pred_sim[match_index[:, 0]] / self.student_temp, dim=-1
                ),
                dim=-1,
            )
            mask_loss = torch_scatter.segment_coo(
                mask_loss,
                index=mask_global_point_student.batch[match_index[:, 0]],
                reduce="mean",
            ).mean()
            metrics["mask_loss"] = mask_loss
            loss_terms.append(mask_loss * self.mask_loss_weight)

        if self.roll_mask_loss_weight > 0:
            roll_global_point_teacher = self.roll_point(global_point_teacher)
            with torch.no_grad():
                match_index = self.match_neighbour(
                    mask_global_point_student.origin_coord,
                    mask_global_point_student.offset,
                    roll_global_point_teacher.origin_coord,
                    roll_global_point_teacher.offset,
                )
                roll_mask_target_sim = self.sinkhorn_knopp(
                    roll_global_point_teacher.feat[match_index[:, 1]],
                    self.teacher_temp,
                )
            roll_mask_loss = -torch.sum(
                roll_mask_target_sim
                * F.log_softmax(
                    mask_pred_sim[match_index[:, 0]] / self.student_temp, dim=-1
                ),
                dim=-1,
            )
            roll_mask_loss = torch_scatter.segment_coo(
                roll_mask_loss,
                index=mask_global_point_student.batch[match_index[:, 0]],
                reduce="mean",
            ).mean()
            metrics["roll_mask_loss"] = roll_mask_loss
            loss_terms.append(roll_mask_loss * self.roll_mask_loss_weight)
        return metrics, loss_terms, mask_global_point_student

    def _compute_unmask_loss(
        self,
        global_point_teacher: Point,
        global_feat: torch.Tensor,
        local_point: Point,
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor], Optional[Point]]:
        metrics: Dict[str, torch.Tensor] = {}
        loss_terms: List[torch.Tensor] = []
        if self.unmask_loss_weight <= 0:
            return metrics, loss_terms, None

        with torch.no_grad():
            global_point_teacher.feat = self.teacher.unmask_head(global_feat)
        local_point_student = self.student.backbone(local_point)
        local_point_student = self.up_cast(local_point_student)
        unmask_pred_sim = self.student.unmask_head(local_point_student.feat)
        with torch.no_grad():
            principal_view_mask = global_point_teacher.batch % self.num_global_view == 0
            principal_view_batch = (
                global_point_teacher.batch[principal_view_mask] // self.num_global_view
            )
            match_index = self.match_neighbour(
                local_point_student.origin_coord,
                local_point_student.offset[self.num_local_view - 1 :: self.num_local_view],
                global_point_teacher.origin_coord[principal_view_mask],
                batch2offset(principal_view_batch),
            )
            unmask_target_sim = self.sinkhorn_knopp(
                global_point_teacher.feat[principal_view_mask][match_index[:, 1]],
                self.teacher_temp,
            )
        unmask_loss = -torch.sum(
            unmask_target_sim
            * F.log_softmax(
                unmask_pred_sim[match_index[:, 0]] / self.student_temp, dim=-1
            ),
            dim=-1,
        )
        unmask_loss = torch_scatter.segment_coo(
            unmask_loss,
            index=local_point_student.batch[match_index[:, 0]],
            reduce="mean",
        ).mean()
        metrics["unmask_loss"] = unmask_loss
        loss_terms.append(unmask_loss * self.unmask_loss_weight)
        return metrics, loss_terms, local_point_student

    def _compute_laplacian_loss(
        self,
        mask_global_point_student: Optional[Point],
        local_point_student: Optional[Point],
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        metrics: Dict[str, torch.Tensor] = {}
        loss_terms: List[torch.Tensor] = []
        if self.laplacian_loss_weight <= 0:
            return metrics, loss_terms

        if mask_global_point_student is not None:
            laplacian_loss = self.compute_laplacian_smoothness_loss(
                mask_global_point_student.feat,
                mask_global_point_student.coord,
                mask_global_point_student.offset,
            )
            metrics["laplacian_loss"] = laplacian_loss
            loss_terms.append(laplacian_loss * self.laplacian_loss_weight)
        elif local_point_student is not None:
            laplacian_loss = self.compute_laplacian_smoothness_loss(
                local_point_student.feat,
                local_point_student.coord,
                local_point_student.offset,
            )
            metrics["laplacian_loss"] = laplacian_loss
            loss_terms.append(laplacian_loss * self.laplacian_loss_weight)
        return metrics, loss_terms

    def _perturb_global_point(self, data_dict: Dict[str, torch.Tensor]) -> Point:
        return Point(
            feat=data_dict["global_feat"],
            coord=data_dict["global_coord"]
            + torch.randn_like(data_dict["global_coord"]) * self.consistency_augment_strength,
            origin_coord=data_dict["global_origin_coord"],
            offset=data_dict["global_offset"],
            grid_size=data_dict["grid_size"][0],
        )

    def _compute_consistency_regularization(
        self,
        data_dict: Dict[str, torch.Tensor],
        mask_global_point_student: Optional[Point],
        local_point_student: Optional[Point],
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        metrics: Dict[str, torch.Tensor] = {}
        loss_terms: List[torch.Tensor] = []
        if self.consistency_loss_weight <= 0 or self.num_global_view <= 1:
            return metrics, loss_terms

        with torch.no_grad():
            teacher_point = self.teacher.backbone(self._perturb_global_point(data_dict))
            teacher_point = self.up_cast(teacher_point)

        if mask_global_point_student is not None:
            consistency_loss = self.compute_consistency_loss(
                teacher_point, mask_global_point_student
            )
            metrics["consistency_loss"] = consistency_loss
            loss_terms.append(consistency_loss * self.consistency_loss_weight)
        elif local_point_student is not None:
            consistency_loss = self.compute_consistency_loss(teacher_point, local_point_student)
            metrics["consistency_loss"] = consistency_loss
            loss_terms.append(consistency_loss * self.consistency_loss_weight)
        return metrics, loss_terms

    @staticmethod
    def _reduce_scalar_dict(result_dict: Dict[str, torch.Tensor]) -> None:
        if get_world_size() <= 1:
            return
        for value in result_dict.values():
            if torch.is_tensor(value) and value.dim() == 0:
                dist.all_reduce(value, op=dist.ReduceOp.AVG)

    def forward(
        self, data_dict: Dict[str, torch.Tensor], return_point: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            return_point: If False, return pretraining losses.
                If True, return teacher features for extraction.
        """
        if return_point:
            point = self.teacher.backbone(data_dict)
            point = self.up_cast(point)
            return dict(point=point)

        with torch.no_grad():
            global_point, mask_global_point, local_point = self._prepare_points(data_dict)
            global_point_ = self.teacher.backbone(global_point)
            global_point_ = self.up_cast(global_point_)
            global_feat = global_point_.feat

        result_dict: Dict[str, torch.Tensor] = {}
        loss_terms: List[torch.Tensor] = []

        mask_metrics, mask_terms, mask_global_point_ = self._compute_mask_losses(
            global_point_, global_feat, mask_global_point
        )
        result_dict.update(mask_metrics)
        loss_terms.extend(mask_terms)

        unmask_metrics, unmask_terms, local_point_ = self._compute_unmask_loss(
            global_point_, global_feat, local_point
        )
        result_dict.update(unmask_metrics)
        loss_terms.extend(unmask_terms)

        lap_metrics, lap_terms = self._compute_laplacian_loss(mask_global_point_, local_point_)
        result_dict.update(lap_metrics)
        loss_terms.extend(lap_terms)

        cons_metrics, cons_terms = self._compute_consistency_regularization(
            data_dict, mask_global_point_, local_point_
        )
        result_dict.update(cons_metrics)
        loss_terms.extend(cons_terms)

        result_dict["loss"] = sum(loss_terms)
        self._reduce_scalar_dict(result_dict)
        return result_dict
