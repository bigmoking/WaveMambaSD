"""Benchmark detector inference latency and throughput on one GPU."""

import argparse
import json
import os.path as osp
import statistics
import time

import torch
from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument(
        'checkpoint',
        help='Checkpoint path, or "none" to benchmark the current structure '
             'with random initialization.')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--output', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.launcher = 'none'
    use_checkpoint = args.checkpoint.lower() not in ('none', 'null', '-')
    cfg.load_from = args.checkpoint if use_checkpoint else None
    if not use_checkpoint and 'backbone' in cfg.model:
        # Avoid downloading the backbone pretrain file for a pure speed test.
        cfg.model.backbone.init_cfg = None
    cfg.work_dir = osp.join('work_dirs', 'fps_benchmark')
    cfg.test_dataloader.batch_size = args.batch_size
    cfg.test_dataloader.num_workers = args.num_workers
    cfg.test_dataloader.persistent_workers = args.num_workers > 0

    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    model = runner.model.cuda().eval()
    data_loader = runner.test_dataloader

    times = []
    measured_images = 0
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for index, data in enumerate(data_loader):
            if index >= args.warmup + args.iterations:
                break
            torch.cuda.synchronize()
            start = time.perf_counter()
            predictions = model.test_step(data)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if index >= args.warmup:
                times.append(elapsed)
                measured_images += len(predictions)
            del predictions

    total_time = sum(times)
    result = {
        'checkpoint': (osp.abspath(args.checkpoint)
                       if use_checkpoint else None),
        'gpu': torch.cuda.get_device_name(),
        'batch_size': args.batch_size,
        'input_scale': list(cfg.get('img_scale', (512, 512))),
        'warmup_iterations': args.warmup,
        'measured_iterations': len(times),
        'measured_images': measured_images,
        'mean_batch_latency_ms': 1000 * statistics.mean(times),
        'median_batch_latency_ms': 1000 * statistics.median(times),
        'mean_latency_ms_per_image': 1000 * total_time / measured_images,
        'fps': measured_images / total_time,
        'peak_gpu_memory_mb': torch.cuda.max_memory_allocated() / 1024**2,
        'timing_scope': 'preprocess + model forward + detection postprocess; excludes dataloader',
        'precision': 'FP32',
    }
    print('FPS_RESULT ' + json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as file:
            json.dump(result, file, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
