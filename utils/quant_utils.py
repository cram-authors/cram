import torch
import pdb

@torch.no_grad()
def one_shot_quant(model, q, only_weights=False, per_channel=False):
    scale_min = -2**(q-1)
    scale_max = 2**(q-1) - 1
    for p in model.parameters():
        if only_weights and (len(p.shape)<2):
            continue
        param_min = min(p.min().item(), 0.0)
        param_max = max(p.max().item(), 0.0)

        # calc scale, zero point is 0 for symmetric
        scale = (
                (param_max - param_min) / (2**q - 1.)
                    if param_max != param_min
                    else 1.0
                )

        # emulate quant
        p.div_(scale)
        p.round_()
        p.clip_(scale_min, scale_max)
        # emulate dequant
        p.mul_(scale)


@torch.no_grad()
def one_shot_quant_new(model, bits, only_weights=True, per_channel=True):
    for p in model.parameters():
        if only_weights and (len(p.shape)<2):
            continue
        weight_quant(p, bits=bits, per_channel=per_channel)
        #p.copy_(weight_quant(p, bits=bits, per_channel=per_channel))
        

def weight_quant(w, bits=8, per_channel=False):
    dev = w.device
    maxq = torch.tensor(2**bits - 1)
    shape = w.shape
    if per_channel:
        w = w.flatten(1)
    else:
        w = w.flatten().unsqueeze(0)

    tmp = torch.zeros(w.shape[0], device=dev)
    wmin = torch.minimum(w.min(1)[0], tmp)
    wmax = torch.maximum(w.max(1)[0], tmp)

    wmax = torch.maximum(torch.abs(wmin), wmax)
    tmp = wmin<0
    if torch.any(tmp):
        wmin[tmp] = -wmax[tmp]
    tmp = (wmin==0) & (wmax==0)
    wmin[tmp] = -1
    wmax[tmp] = +1
    
    scale = (wmax - wmin) / maxq
    if not per_channel:
        tmp = shape[0]
        scale = scale.repeat(tmp)

    new_shape = [-1] + [1] * (len(shape)-1)
    scale = scale.reshape(new_shape)
    w = w.reshape(shape)
    if torch.all(scale!=0):
        w.div_(scale)
        w.round_()
        w.clamp_(-(maxq + 1)/2, 0.5 * (maxq + 1) - 1)
        w.mul_(scale)
    return w
