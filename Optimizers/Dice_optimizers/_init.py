
from .DiceSGD import DPOptimizer_Dice




def get_optimizer_class(
    clipping: str, distributed: bool, grad_sample_mode: str = None, Dice: bool = False
):
        return DPOptimizer_Dice

    #     if grad_sample_mode == "ghost":
    #         if clipping == "flat" and distributed is False:
    #             return KF_DPOptimizerFastGradientClipping
    #         elif clipping == "flat" and distributed is True:
    #             return KF_DistributedDPOptimizerFastGradientClipping
    #         else:
    #
    #             raise ValueError(err_str)
    #     elif clipping == "flat" and distributed is False:
    #         return KF_DPOptimizer
    #     elif clipping == "flat" and distributed is True:
    #         return KF_DistributedDPOptimizer
    #     elif clipping == "per_layer" and distributed is False:
    #         return KF_DPPerLayerOptimizer
    #     elif clipping == "per_layer" and distributed is True:
    #         if grad_sample_mode == "hooks":
    #             return KF_DistributedPerLayerOptimizer
    #         elif grad_sample_mode == "ew":
    #             return KF_SimpleDistributedPerLayerOptimizer
    #         else:
    #             raise ValueError(f"Unexpected grad_sample_mode: {grad_sample_mode}")
    #     elif clipping == "adaptive" and distributed is False:
    #         return KF_AdaClipDPOptimizer
    # elif grad_sample_mode == "ghost":
    #     if clipping == "flat" and distributed is False:
    #         return DPOptimizerFastGradientClipping
    #     elif clipping == "flat" and distributed is True:
    #         return DistributedDPOptimizerFastGradientClipping
    #     else:
    #         err_str = "Unsupported combination of parameters."
    #         err_str+= f"Clipping: {clipping} and grad_sample_mode: {grad_sample_mode}"
    #         raise ValueError(
    #             err_str
    #         )
    # elif clipping == "flat" and distributed is False:
    #     return DPOptimizer
    # elif clipping == "flat" and distributed is True:
    #     return DistributedDPOptimizer
    # elif clipping == "per_layer" and distributed is False:
    #     return DPPerLayerOptimizer
    # elif clipping == "per_layer" and distributed is True:
    #     if grad_sample_mode == "hooks":
    #         return DistributedPerLayerOptimizer
    #     elif grad_sample_mode == "ew":
    #         return SimpleDistributedPerLayerOptimizer
    #     else:
    #         raise ValueError(f"Unexpected grad_sample_mode: {grad_sample_mode}")
    # elif clipping == "adaptive" and distributed is False:
    #     return AdaClipDPOptimizer
    # raise ValueError(
    #     f"Unexpected optimizer parameters. Clipping: {clipping}, distributed: {distributed}"
    # )