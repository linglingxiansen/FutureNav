#!/usr/bin/env python3
"""
FutureNav R2R VLN-CE evaluation entry point (habitat 0.1.7 compatible).
"""

import sys
import os
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

import argparse
import json
import time

import numpy as np
from habitat import Env
from habitat.datasets import make_dataset
from tqdm import trange
from VLN_CE.vlnce_baselines.config.default import get_config

from agent import FutureNav_R2R_Agent


def main():
    parser = argparse.ArgumentParser(description="FutureNav R2R VLN-CE Evaluation")
    parser.add_argument("--exp-config", type=str, required=True, help="Habitat experiment config yaml")
    parser.add_argument("--split-num", type=int, required=True, help="Number of parallel splits")
    parser.add_argument("--split-id", type=int, required=True, help="This split's ID")
    parser.add_argument("--model-path", type=str, required=True, help="Model checkpoint directory")
    parser.add_argument("--result-path", type=str, required=True, help="Directory to save results")
    parser.add_argument("--exp-save", type=str, default="data", help="'video-data' to save maps")
    parser.add_argument("--num-history", type=int, default=8, help="Max history frames")
    parser.add_argument("--max-steps", type=int, default=400, help="Max steps per episode")
    parser.add_argument("--max-episodes", type=int, default=0, help="Max episodes to eval (0=all)")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID for rendering")
    args = parser.parse_args()

    config = get_config(args.exp_config)
    # Override GPU device ID for Habitat-sim rendering
    config.defrost()
    config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.gpu_id
    config.freeze()
    dataset = make_dataset(id_dataset=config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
    dataset.episodes.sort(key=lambda ep: ep.episode_id)
    np.random.seed(42)

    # Optionally limit number of episodes
    if args.max_episodes > 0:
        total = len(dataset.episodes)
        indices = np.random.choice(total, min(args.max_episodes, total), replace=False)
        indices.sort()
        dataset.episodes = [dataset.episodes[i] for i in indices]
        print(f"Randomly selected {len(dataset.episodes)}/{total} episodes for evaluation")

    dataset_split = dataset.get_splits(args.split_num)[args.split_id]
    evaluate(config, args.split_id, dataset_split, args)


def evaluate(config, split_id, dataset, args):
    print(f"[chunk{split_id}] GPU_DEVICE_ID={config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID}, --gpu-id={args.gpu_id}")

    # Detect already completed episodes (before creating env to avoid abort on close)
    log_dir = os.path.join(args.result_path, "log")
    completed_ids = set()
    if os.path.exists(log_dir):
        for f in os.listdir(log_dir):
            if f.startswith("stats_") and f.endswith(".json"):
                ep_id = f[len("stats_"):-len(".json")]
                completed_ids.add(ep_id)
    if completed_ids:
        print(f"[chunk{split_id}] Found {len(completed_ids)} completed episodes, will skip them")

    # Check if all episodes in this split are already done
    all_ep_ids = set(str(ep.episode_id) for ep in dataset.episodes)
    remaining = all_ep_ids - completed_ids
    if not remaining:
        print(f"[chunk{split_id}] All {len(all_ep_ids)} episodes already completed, nothing to do.")
        return

    print(f"[chunk{split_id}] {len(remaining)} episodes remaining to run")

    env = Env(config.TASK_CONFIG, dataset)
    require_map = "video" in (args.exp_save or "")

    agent = FutureNav_R2R_Agent(
        checkpoint_path=args.model_path,
        result_path=args.result_path,
        require_map=require_map,
        max_history_frames=args.num_history,
        device=f"cuda:{args.gpu_id}",
    )

    num_episodes = len(env.episodes)
    EARLY_STOP_ROTATION = config.EVAL.get("EARLY_STOP_ROTATION", 25)
    EARLY_STOP_STEPS = config.EVAL.get("EARLY_STOP_STEPS", args.max_steps)
    target_key = {"distance_to_goal", "success", "spl", "ndtw", "path_length", "oracle_success"}

    all_results = []

    # Debug timing stats
    total_render_time = 0.0
    total_infer_time = 0.0
    total_steps = 0

    skipped = 0
    for ep_idx in trange(num_episodes, desc=f"split-{split_id}"):
        # Check if this episode is already done
        ep_id_check = str(env.episodes[ep_idx % num_episodes].episode_id) if ep_idx < num_episodes else None
        obs = env.reset()
        current_ep_id = str(env.current_episode.episode_id)
        if current_ep_id in completed_ids:
            skipped += 1
            # Need to end the episode by stepping STOP
            obs = env.step({"action": 0})
            continue

        ep_start = time.time()
        agent.reset()
        iter_step = 0
        continuous_rotation_count = 0
        last_dtg = 999

        while not env.episode_over:
            info = env.get_metrics()

            if info["distance_to_goal"] != last_dtg:
                last_dtg = info["distance_to_goal"]
                continuous_rotation_count = 0
            else:
                continuous_rotation_count += 1

            # Time model inference
            t0 = time.time()
            action = agent.act(obs, info, env.current_episode.episode_id)
            t1 = time.time()
            infer_elapsed = t1 - t0
            total_infer_time += infer_elapsed

            # Early stop if stuck rotating or too many steps
            if continuous_rotation_count > EARLY_STOP_ROTATION or iter_step > EARLY_STOP_STEPS:
                action = {"action": 0}

            # Time habitat sim step (rendering)
            t2 = time.time()
            iter_step += 1
            obs = env.step(action)
            t3 = time.time()
            render_elapsed = t3 - t2
            total_render_time += render_elapsed
            total_steps += 1

            # Print per-step timing for first 3 episodes
            if ep_idx < 3:
                print(f"  [chunk{split_id} ep{ep_idx} step{iter_step}] infer={infer_elapsed:.3f}s render={render_elapsed:.3f}s")

        ep_elapsed = time.time() - ep_start
        if ep_idx < 10:
            print(f"[chunk{split_id} Episode {ep_idx}] steps={iter_step} total={ep_elapsed:.2f}s "
                  f"avg_infer={total_infer_time/max(total_steps,1):.3f}s "
                  f"avg_render={total_render_time/max(total_steps,1):.3f}s")

        info = env.get_metrics()
        result_dict = {k: info[k] for k in target_key if k in info}
        result_dict["id"] = env.current_episode.episode_id
        result_dict["steps"] = iter_step
        all_results.append(result_dict)

        if "data" in args.exp_save:
            log_dir = os.path.join(args.result_path, "log")
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, f"stats_{env.current_episode.episode_id}.json"), "w") as f:
                json.dump(result_dict, f, indent=4)

    print(f"[chunk{split_id}] Done. Skipped {skipped} already-completed episodes, ran {len(all_results)} new episodes.")

    # Load previously completed results and merge
    all_results_merged = []
    if os.path.exists(log_dir):
        for f_name in os.listdir(log_dir):
            if f_name.startswith("stats_") and f_name.endswith(".json"):
                with open(os.path.join(log_dir, f_name)) as f_read:
                    all_results_merged.append(json.load(f_read))
    else:
        all_results_merged = all_results

    # Save summary for this split
    summary_path = os.path.join(args.result_path, f"summary_split_{split_id}.json")
    summary = {
        "split_id": split_id,
        "num_episodes": num_episodes,
        "skipped": skipped,
        "new_episodes": len(all_results),
        "total_completed": len(all_results_merged),
        "results": all_results_merged,
    }
    if all_results:
        summary["avg_success"] = np.mean([r.get("success", 0) for r in all_results])
        summary["avg_spl"] = np.mean([r.get("spl", 0) for r in all_results])
        summary["avg_dtg"] = np.mean([r.get("distance_to_goal", 0) for r in all_results])
        summary["avg_path_length"] = np.mean([r.get("path_length", 0) for r in all_results])
        summary["avg_oracle_success"] = np.mean([r.get("oracle_success", 0) for r in all_results])
        print(f"\n[Split {split_id}] SR={summary['avg_success']:.4f} SPL={summary['avg_spl']:.4f} "
              f"DTG={summary['avg_dtg']:.2f} PL={summary['avg_path_length']:.2f} "
              f"OS={summary['avg_oracle_success']:.4f}")

    # Print overall timing summary
    print(f"\n{'='*60}")
    print(f"[TIMING SUMMARY] split={split_id}")
    print(f"  Total steps: {total_steps}")
    print(f"  Total render time: {total_render_time:.2f}s (avg {total_render_time/max(total_steps,1)*1000:.1f}ms/step)")
    print(f"  Total infer time:  {total_infer_time:.2f}s (avg {total_infer_time/max(total_steps,1)*1000:.1f}ms/step)")
    print(f"  Render fraction:   {total_render_time/(total_render_time+total_infer_time+1e-9)*100:.1f}%")
    print(f"  Infer fraction:    {total_infer_time/(total_render_time+total_infer_time+1e-9)*100:.1f}%")
    print(f"{'='*60}\n")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Split {split_id} results saved to: {summary_path}")

    env.close()


if __name__ == "__main__":
    main()
