import copy
from typing import Tuple



def stats(
    cfg,
    input_shape: Tuple = (1, 3, 640, 640),
) -> Tuple[int, dict]:
    base_size = cfg.train_dataloader.collate_fn.base_size
    input_shape = (1, 3, base_size, base_size)

    model_for_info = copy.deepcopy(cfg.model).deploy()
    params = sum(p.numel() for p in model_for_info.parameters())

    try:
        from calflops import calculate_flops

        flops, macs, _ = calculate_flops(
            model=model_for_info,
            input_shape=input_shape,
            output_as_string=True,
            output_precision=4,
            print_detailed=False,
        )
        model_stats = {"Model FLOPs:%s   MACs:%s   Params:%s" % (flops, macs, params)}
    except (ImportError, RuntimeError) as exc:
        model_stats = {"Model FLOPs:unavailable   MACs:unavailable   Params:%s   profiler_error:%s" % (params, str(exc).split("\n")[0])}

    del model_for_info

    return params, model_stats
