import math
import torch
import torch.nn as nn


def normalize_img(img, image_mean, image_std):
    img = img / 255.0
    img = img.permute(2,0,1)
    img = (img - image_mean[:,None,None]) / image_std[:,None,None]
    
    return img

class ConvNormActBlock(torch.nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        norm_layer=None,
        activation_layer=None,
    ):
        super().__init__()
        
        padding = (kernel_size - 1) // 2
        
        bias = norm_layer is None
        
        self.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=bias,
            )
        )
        if norm_layer is not None:
            self.append(norm_layer(out_channels))
        
        if activation_layer is not None:
            self.append(activation_layer())


def decode_boxes(rel_codes, boxes, weights=(1.0, 1.0, 1.0, 1.0), bbox_xform_clip=math.log(1000.0 / 16)):

    wx, wy, ww, wh = weights
    tx = rel_codes[:, 0] / wx
    ty = rel_codes[:, 1] / wy
    tw = rel_codes[:, 2] / ww
    th = rel_codes[:, 3] / wh

    tw = torch.clamp(tw, max=bbox_xform_clip)
    th = torch.clamp(th, max=bbox_xform_clip)

    boxes = boxes.to(rel_codes.dtype)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    pred_ctr_x = ctr_x + tx * widths
    pred_ctr_y = ctr_y + ty * heights
    pred_w = widths * torch.exp(tw)
    pred_h = heights * torch.exp(th)

    decoded_boxes = torch.zeros_like(rel_codes)
    decoded_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w  # x1
    decoded_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h  # y1
    decoded_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w  # x2
    decoded_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h  # y2

    return decoded_boxes
