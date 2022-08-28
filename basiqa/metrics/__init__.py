from copy import deepcopy

from utils.registry import METRIC_REGISTRY
from .niqe import calculate_niqe
from .psnr_ssim import calculate_psnr, calculate_ssim
from .iqa_metrics import calculate_l1,calculate_plcc,calculate_srcc
from .classi_metrics import get_confus_matrix,calculate_acc,calculate_f1,calculate_p,calculate_r

__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_niqe','calculate_l1']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
