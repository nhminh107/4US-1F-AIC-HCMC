"""Kiến trúc PyTorch của TransNetV2 dùng để phát hiện ranh giới shot (shot boundary).

Được vendor (copy có chỉnh sửa) từ ``soCzech/TransNetV2`` (giấy phép MIT,
Copyright (c) 2020 Tomas Soucek), file
``inference-pytorch/transnetv2_pytorch.py``:
https://github.com/soCzech/TransNetV2

Chỉ vendor phần định nghĩa layer và forward pass. Tên layer, tên tham số,
và shape tensor phải giữ **giống hệt** bản gốc vì trọng số đã pretrain
(convert từ checkpoint TensorFlow gốc, xem ``transnetv2-pytorch-weights.pth``
ở mục 2.3 của ``module_shot_keyframe.md``) sẽ được nạp vào đúng module này
qua ``nn.Module.load_state_dict``. Đổi tên một layer hay reshape một tensor
ở đây sẽ khiến việc nạp trọng số lỗi (hoặc tệ hơn là nạp sai âm thầm).

Không dùng module này để train; đây là bản chỉ phục vụ inference, đúng như
dự án gốc.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as functional

# Kích thước ảnh đầu vào (rộng, cao) mà network kỳ vọng cho mỗi frame.
INPUT_WIDTH = 48
INPUT_HEIGHT = 27


class TransNetV2(nn.Module):
    """Bộ phát hiện ranh giới shot: một cửa sổ frame ngắn -> logits cho từng frame."""

    def __init__(
        self,
        F: int = 16,
        L: int = 3,
        S: int = 2,
        D: int = 1024,
        use_many_hot_targets: bool = True,
        use_frame_similarity: bool = True,
        use_color_histograms: bool = True,
        use_mean_pooling: bool = False,
        dropout_rate: float | None = 0.5,
        use_convex_comb_reg: bool = False,  # bản port này không hỗ trợ
        use_resnet_features: bool = False,  # bản port này không hỗ trợ
        use_resnet_like_top: bool = False,  # bản port này không hỗ trợ
        frame_similarity_on_last_layer: bool = False,  # bản port này không hỗ trợ
    ) -> None:
        super().__init__()

        if use_resnet_features or use_resnet_like_top or use_convex_comb_reg or frame_similarity_on_last_layer:
            raise NotImplementedError(
                "This PyTorch port only supports the default TransNetV2 "
                "configuration used to train the public checkpoint."
            )

        self.SDDCNN = nn.ModuleList(
            [StackedDDCNNV2(in_filters=3, n_blocks=S, filters=F, stochastic_depth_drop_prob=0.0)]
            + [
                StackedDDCNNV2(in_filters=(F * 2 ** (i - 1)) * 4, n_blocks=S, filters=F * 2**i)
                for i in range(1, L)
            ]
        )

        self.frame_sim_layer = (
            FrameSimilarity(
                sum((F * 2**i) * 4 for i in range(L)),
                lookup_window=101,
                output_dim=128,
                similarity_dim=128,
                use_bias=True,
            )
            if use_frame_similarity
            else None
        )
        self.color_hist_layer = (
            ColorHistograms(lookup_window=101, output_dim=128) if use_color_histograms else None
        )

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else None

        output_dim = ((F * 2 ** (L - 1)) * 4) * 3 * 6  # 3x6 là kích thước không gian sau 3 tầng pooling
        if use_frame_similarity:
            output_dim += 128
        if use_color_histograms:
            output_dim += 128

        self.fc1 = nn.Linear(output_dim, D)
        self.cls_layer1 = nn.Linear(D, 1)
        self.cls_layer2 = nn.Linear(D, 1) if use_many_hot_targets else None

        self.use_mean_pooling = use_mean_pooling
        self.eval()

    def forward(self, inputs: torch.Tensor):
        """Chạy inference trên một batch các cửa sổ frame có độ dài cố định.

        ``inputs`` phải là kiểu ``uint8`` với shape ``[batch, frames, 27, 48, 3]``
        (RGB, không phải BGR) — đúng contract mà trọng số pretrain yêu cầu.
        Trả về logits ranh giới cho từng frame, và nếu head many-hot được bật
        thì trả thêm dict phụ ``{"many_hot": ...}``.
        """

        if not (isinstance(inputs, torch.Tensor) and list(inputs.shape[2:]) == [27, 48, 3] and inputs.dtype == torch.uint8):
            raise ValueError(
                "TransNetV2 expects a uint8 tensor of shape "
                "[batch, frames, 27, 48, 3]; got "
                f"shape={tuple(inputs.shape)} dtype={inputs.dtype}"
            )

        # uint8 [B, T, H, W, 3] -> float [B, 3, T, H, W], chuẩn hoá về [0, 1].
        x = inputs.permute([0, 4, 1, 2, 3]).float()
        x = x.div_(255.0)

        block_features = []
        for block in self.SDDCNN:
            x = block(x)
            block_features.append(x)

        if self.use_mean_pooling:
            x = torch.mean(x, dim=[3, 4])
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 3, 4, 1)
            x = x.reshape(x.shape[0], x.shape[1], -1)

        if self.frame_sim_layer is not None:
            x = torch.cat([self.frame_sim_layer(block_features), x], 2)

        if self.color_hist_layer is not None:
            x = torch.cat([self.color_hist_layer(inputs), x], 2)

        x = self.fc1(x)
        x = functional.relu(x)

        if self.dropout is not None:
            x = self.dropout(x)

        one_hot = self.cls_layer1(x)

        if self.cls_layer2 is not None:
            return one_hot, {"many_hot": self.cls_layer2(x)}
        return one_hot


class StackedDDCNNV2(nn.Module):
    """Một tầng downsampling: chồng các block conv giãn nở (dilated) + pooling."""

    def __init__(
        self,
        in_filters: int,
        n_blocks: int,
        filters: int,
        shortcut: bool = True,
        pool_type: str = "avg",
        stochastic_depth_drop_prob: float = 0.0,
    ) -> None:
        super().__init__()

        assert pool_type in ("max", "avg")

        self.shortcut = shortcut
        self.DDCNN = nn.ModuleList(
            [
                DilatedDCNNV2(
                    in_filters if i == 1 else filters * 4,
                    filters,
                    activation=functional.relu if i != n_blocks else None,
                )
                for i in range(1, n_blocks + 1)
            ]
        )
        self.pool = (
            nn.MaxPool3d(kernel_size=(1, 2, 2))
            if pool_type == "max"
            else nn.AvgPool3d(kernel_size=(1, 2, 2))
        )
        self.stochastic_depth_drop_prob = stochastic_depth_drop_prob

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        shortcut = None

        for block in self.DDCNN:
            x = block(x)
            if shortcut is None:
                shortcut = x

        x = functional.relu(x)

        if self.shortcut is not None:
            if self.stochastic_depth_drop_prob != 0.0:
                if self.training:
                    if random.random() < self.stochastic_depth_drop_prob:
                        x = shortcut
                    else:
                        x = x + shortcut
                else:
                    x = (1 - self.stochastic_depth_drop_prob) * x + shortcut
            else:
                x = x + shortcut

        return self.pool(x)


class DilatedDCNNV2(nn.Module):
    """Bốn nhánh convolution (2+1)D giãn nở song song, ghép nối rồi chuẩn hoá."""

    def __init__(
        self,
        in_filters: int,
        filters: int,
        batch_norm: bool = True,
        activation=None,
    ) -> None:
        super().__init__()

        self.Conv3D_1 = Conv3DConfigurable(in_filters, filters, 1, use_bias=not batch_norm)
        self.Conv3D_2 = Conv3DConfigurable(in_filters, filters, 2, use_bias=not batch_norm)
        self.Conv3D_4 = Conv3DConfigurable(in_filters, filters, 4, use_bias=not batch_norm)
        self.Conv3D_8 = Conv3DConfigurable(in_filters, filters, 8, use_bias=not batch_norm)

        self.bn = nn.BatchNorm3d(filters * 4, eps=1e-3) if batch_norm else None
        self.activation = activation

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = torch.cat(
            [
                self.Conv3D_1(inputs),
                self.Conv3D_2(inputs),
                self.Conv3D_4(inputs),
                self.Conv3D_8(inputs),
            ],
            dim=1,
        )

        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Conv3DConfigurable(nn.Module):
    """Một convolution (2+1)D: conv không gian 3x3 rồi đến conv thời gian giãn nở 3x1x1."""

    def __init__(
        self,
        in_filters: int,
        filters: int,
        dilation_rate: int,
        separable: bool = True,
        use_bias: bool = True,
    ) -> None:
        super().__init__()

        if separable:
            # Convolution (2+1)D, xem https://arxiv.org/pdf/1711.11248.pdf
            conv1 = nn.Conv3d(
                in_filters, 2 * filters, kernel_size=(1, 3, 3),
                dilation=(1, 1, 1), padding=(0, 1, 1), bias=False,
            )
            conv2 = nn.Conv3d(
                2 * filters, filters, kernel_size=(3, 1, 1),
                dilation=(dilation_rate, 1, 1), padding=(dilation_rate, 0, 0), bias=use_bias,
            )
            self.layers = nn.ModuleList([conv1, conv2])
        else:
            conv = nn.Conv3d(
                in_filters, filters, kernel_size=3,
                dilation=(dilation_rate, 1, 1), padding=(dilation_rate, 1, 1), bias=use_bias,
            )
            self.layers = nn.ModuleList([conv])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class FrameSimilarity(nn.Module):
    """Độ tương đồng (self-similarity) giữa các frame đã chiếu (project), trong một cửa sổ lookup."""

    def __init__(
        self,
        in_filters: int,
        similarity_dim: int = 128,
        lookup_window: int = 101,
        output_dim: int = 128,
        use_bias: bool = False,
    ) -> None:
        super().__init__()

        assert lookup_window % 2 == 1, "`lookup_window` must be an odd integer"

        self.projection = nn.Linear(in_filters, similarity_dim, bias=use_bias)
        self.fc = nn.Linear(lookup_window, output_dim)
        self.lookup_window = lookup_window

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        x = torch.cat([torch.mean(features, dim=[3, 4]) for features in inputs], dim=1)
        x = torch.transpose(x, 1, 2)

        x = self.projection(x)
        x = functional.normalize(x, p=2, dim=2)

        return _windowed_self_similarity(x, self.lookup_window, self.fc)


class ColorHistograms(nn.Module):
    """Độ tương đồng giữa histogram màu RGB của từng frame, trong một cửa sổ lookup."""

    def __init__(self, lookup_window: int = 101, output_dim: int | None = None) -> None:
        super().__init__()

        assert lookup_window % 2 == 1, "`lookup_window` must be an odd integer"

        self.fc = nn.Linear(lookup_window, output_dim) if output_dim is not None else None
        self.lookup_window = lookup_window

    @staticmethod
    def compute_color_histograms(frames: torch.Tensor) -> torch.Tensor:
        """Lượng tử hoá pixel RGB của mỗi frame thành histogram 512 bin (8x8x8)."""

        frames = frames.int()

        def get_bin(pixels: torch.Tensor) -> torch.Tensor:
            # Lượng tử hoá mỗi kênh 8-bit xuống còn 3-bit -> một bin trong [0, 511].
            red, green, blue = pixels[:, :, 0] >> 5, pixels[:, :, 1] >> 5, pixels[:, :, 2] >> 5
            return (red << 6) + (green << 3) + blue

        batch_size, time_window, height, width, channels = frames.shape
        assert channels == 3
        frames_flatten = frames.view(batch_size * time_window, height * width, 3)

        binned_values = get_bin(frames_flatten)
        frame_bin_prefix = (torch.arange(0, batch_size * time_window, device=frames.device) << 9).view(-1, 1)
        binned_values = (binned_values + frame_bin_prefix).view(-1)

        histograms = torch.zeros(batch_size * time_window * 512, dtype=torch.int32, device=frames.device)
        histograms.scatter_add_(
            0, binned_values, torch.ones(len(binned_values), dtype=torch.int32, device=frames.device)
        )

        histograms = histograms.view(batch_size, time_window, 512).float()
        return functional.normalize(histograms, p=2, dim=2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.compute_color_histograms(inputs)
        similarities = _windowed_self_similarity(x, self.lookup_window, self.fc)
        return similarities


def _windowed_self_similarity(features: torch.Tensor, lookup_window: int, fc: nn.Linear | None) -> torch.Tensor:
    """Độ tương đồng (dạng cosine) của mỗi frame với các frame lân cận theo thời gian.

    ``features`` có shape ``[batch, time, dim]``. Với mỗi frame, tính độ tương
    đồng dạng tích vô hướng (dot-product) với mọi frame khác nằm trong phạm vi
    ``lookup_window`` frame quanh nó (đệm 0 ở hai đầu chuỗi), tuỳ chọn chiếu
    kết quả qua một layer linear + ReLU.
    """

    batch_size, time_window = features.shape[0], features.shape[1]
    similarities = torch.bmm(features, features.transpose(1, 2))  # [batch, time, time]
    similarities_padded = functional.pad(
        similarities, [(lookup_window - 1) // 2, (lookup_window - 1) // 2]
    )

    batch_indices = (
        torch.arange(0, batch_size, device=features.device)
        .view([batch_size, 1, 1])
        .repeat([1, time_window, lookup_window])
    )
    time_indices = (
        torch.arange(0, time_window, device=features.device)
        .view([1, time_window, 1])
        .repeat([batch_size, 1, lookup_window])
    )
    lookup_indices = (
        torch.arange(0, lookup_window, device=features.device)
        .view([1, 1, lookup_window])
        .repeat([batch_size, time_window, 1])
        + time_indices
    )

    windowed = similarities_padded[batch_indices, time_indices, lookup_indices]
    if fc is not None:
        return functional.relu(fc(windowed))
    return windowed
