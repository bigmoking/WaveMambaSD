"""Evaluate a detector under deterministic SAR speckle perturbations."""

import argparse
import json
import os.path as osp

from mmengine.config import Config
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--enl', type=float, default=None,
                        help='Target ENL. Omit for the clean baseline.')
    parser.add_argument('--seed', type=int, default=20260927)
    parser.add_argument('--work-dir', default=None)
    parser.add_argument('--summary', default=None,
                        help='Optional JSON file containing the returned metrics.')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    setup_cache_size_limit_of_dynamo()
    cfg = Config.fromfile(args.config)
    # A config dumped from a distributed training job may retain
    # ``launcher='pytorch'``. Robustness conditions are independent single-GPU
    # jobs, so do not require torchrun/RANK environment variables.
    cfg.launcher = 'none'
    cfg.load_from = args.checkpoint

    label = 'clean' if args.enl is None else f'enl_{args.enl:g}'
    cfg.work_dir = args.work_dir or osp.join(
        'work_dirs', 'sar_robustness', label)
    if args.batch_size is not None:
        cfg.test_dataloader.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.test_dataloader.num_workers = args.num_workers
        cfg.test_dataloader.persistent_workers = args.num_workers > 0

    if args.enl is not None:
        pipeline = cfg.test_dataloader.dataset.pipeline
        load_index = next(
            i for i, transform in enumerate(pipeline)
            if transform['type'].split('.')[-1] == 'LoadImageFromFile')
        pipeline.insert(load_index + 1, dict(
            type='groksar.AddSpeckleNoise', enl=args.enl, seed=args.seed,
            shared_channels=True, clip=True))

    metrics = Runner.from_cfg(cfg).test()
    result = dict(condition=label, enl=args.enl, seed=args.seed,
                  checkpoint=osp.abspath(args.checkpoint), metrics=metrics)
    summary = args.summary or osp.join(cfg.work_dir, 'metrics.json')
    with open(summary, 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    print('ROBUSTNESS_RESULT ' + json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
