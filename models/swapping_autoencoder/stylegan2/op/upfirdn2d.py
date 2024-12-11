import torch
import torch.nn.functional as F


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):

    out = upfirdn2d_native(
        input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]
    )

    return out


def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    bs, ch, in_h, in_w = input.shape
    minor = 1
    kernel_h, kernel_w = kernel.shape
    out = input.view(-1, in_h, 1, in_w, 1, minor)
    if up_x > 1 or up_y > 1:
        out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])

    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    if pad_x0 > 0 or pad_x1 > 0 or pad_y0 > 0 or pad_y1 > 0:
        out = F.pad(
            out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
        )

    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )

    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)

    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )

    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]
    out = out.view(bs, ch, out.size(1), out.size(2))
    return out
