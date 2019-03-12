import torch


def jaccard_loss(inputs, targets):
    num = targets.size()[0]  # batch size
    inputSize = targets.size()[2]
    m1 = inputs.contiguous().view(num, -1)
    m2 = targets.contiguous().view(num, -1)

    numerator = torch.sum(m1 * m2, 1)
    denominator = torch.sum(m2, 1)

    numerator2 = torch.sum((1 - m1) * (1 - m2), 1)
    denominator2 = torch.sum((1 - m2), 1)
    ratioOfbackgroundPixels = torch.mean(
        denominator2 / inputSize ** 2)  # give larger weights for loss produced by image patches with fewer number of vessel pixels

    denominator3 = torch.sum(m1, 1)
    denominator4 = torch.sum((1 - m1), 1)

    return ((1 - torch.mean(numerator / (denominator + 0.000001))) * ratioOfbackgroundPixels + 1 - torch.mean(
        numerator2 / (denominator2 + 0.000001)) + (1 - ratioOfbackgroundPixels) * (
                        1 - torch.mean(numerator / (denominator3 + 0.000001))) + (
                        1 - torch.mean(numerator2 / (denominator4 + 0.000001)))) / 4

def jaccard_loss_2(inputs, targets):
    num = targets.size()[0]  # batch size
    inputSize = targets.size()[2]
    m1 = inputs.contiguous().view(num, -1)
    m2 = targets.contiguous().view(num, -1)

    numerator = torch.sum(m1 * m2, 1)
    denominator = torch.sum(m2, 1)

    numerator2 = torch.sum((1 - m1) * (1 - m2), 1)
    denominator2 = torch.sum((1 - m2), 1)
    ratioOfbackgroundPixels = torch.mean(
        denominator2 / inputSize ** 2)  # give larger weights for loss produced by image patches with fewer number of vessel pixels

    denominator3 = torch.sum(m1, 1)
    denominator4 = torch.sum((1 - m1), 1)

    return ((1 - torch.mean(numerator / (denominator + 0.000001))) * ratioOfbackgroundPixels + 1 - torch.mean(
        numerator2 / (denominator2 + 0.000001)) + (
                        1 - torch.mean(numerator / (denominator3 + 0.000001))) + (
                        1 - torch.mean(numerator2 / (denominator4 + 0.000001)))) / 4

def dice_loss(pred, target):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1.
    epsilon = 10e-8

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    dice = (2. * intersection + smooth) / (A_sum + B_sum + smooth)
    dice = dice.mean(dim=0)
    dice = torch.clamp(dice, 0, 1.0-epsilon)

    return  1- dice

def dice_hard(pred, target):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1.
    epsilon =  10e-8

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    dice = (2. * intersection + smooth) / (A_sum + B_sum + smooth)
    dice = dice.mean(dim=0)
    dice = torch.clamp(dice, 0, 1.0-epsilon)

    return 1 - (dice)

def dice_coeff(pred, target):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1.
    epsilon = 10e-8

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    dice = (2. * intersection + smooth) / (A_sum + B_sum + smooth)
    dice = dice.mean(dim=0)
    dice = torch.clamp(dice, 0, 1.0-epsilon)

    return  dice

def dice_soft(pred, target, loss_type='sorensen', smooth=1e-5, from_logits=False):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    pred : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    loss_type : string
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    smooth : float
        This small value will be added to the numerator and denominator.
        If both y_pred and y_true are empty, it makes sure dice is 1.
        If either y_pred or y_true are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``,
        then if smooth is very small, dice close to 0 (even the image values lower than the threshold),
        so in this case, higher smooth can have a higher dice.

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`_
    """
    if not from_logits:
        # transform back to logits
        _epsilon = 1e-7
        pred = torch.clamp(pred, _epsilon, 1 - _epsilon)
        pred = torch.log(pred / (1 - pred))

    inse = torch.sum(pred * target)
    if loss_type == 'jaccard':
        l = torch.sum(pred * pred)
        r = torch.sum(target * target)
    elif loss_type == 'sorensen':
        l = torch.sum(pred)
        r = torch.sum(target)
    else:
        raise Exception("Unknow loss_type")

    dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    dice = dice.mean(dim=0)
    return dice
