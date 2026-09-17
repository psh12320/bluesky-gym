"""Record selected development scenarios and verify their saved evaluation outcomes."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from atc.compare import load_evaluation
from atc.metrics import METRICS, SAFETY
from atc.provenance import capture
from atc import submission


def annotated_frame(env, episode, final=False):
    world = env.unwrapped
    frame = env.render()
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError('The original renderer did not return an RGB frame')
    width = 720
    native = Image.fromarray(frame).resize((width, width), Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', (width, width + 108), '#f8fafc')
    canvas.paste(native, (0, 108))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=17)
    except TypeError:
        font = ImageFont.load_default()
    rows = list(world.metrics.values())
    arrived = sum(int(row['waypoint_reached']) for row in rows)
    clean = sum(bool(row['waypoint_reached']) and all(row[key] == 0 for key in SAFETY) for row in rows)
    mean = lambda key: np.mean([row[key] for row in rows])
    elapsed = max(row['flight_time'] for row in rows)
    label = 'Final outcome' if final else 'Policy replay'
    draw.text((12, 10), f'{label} | Test scenario {episode + 1} | Simulation time: {elapsed:.0f} s', fill='#0f172a', font=font)
    draw.text((12, 35), f'Arrivals: {arrived}/{len(rows)} | Arrivals without safety violations: {clean}/{len(rows)}', fill='#0f172a', font=font)
    draw.text((12, 60), f'Mean seconds per aircraft: conflict {mean("intrusion_time"):.1f} | restricted {mean("time_in_restricted_area"):.1f} | outside {mean("time_outside_sector"):.1f}', fill='#334155', font=font)
    draw.text((12, 84), 'Gray: restricted airspace | Green: sector boundary | Separation: 5 NM', fill='#64748b', font=font)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True, help='Existing development evaluation prefix')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--episodes', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-motion-frames', type=int, default=150)
    parser.add_argument('--fps', type=int, default=10)
    args = parser.parse_args()
    metadata, records = load_evaluation(args.reference)
    if metadata['seed'] != 2026 or metadata.get('inference_mode') != 'per_aircraft':
        parser.error('Record existing per-aircraft development evaluations on seed 2026')
    if not args.model.is_file() or hashlib.sha256(args.model.read_bytes()).hexdigest() != metadata.get('model_sha256'):
        parser.error('Checkpoint hash differs from the reference evaluation')
    if not args.episodes or len(set(args.episodes)) != len(args.episodes) or min(args.episodes) < 0 or max(args.episodes) >= metadata['episodes']:
        parser.error('Use distinct episode indices available in the reference')
    if args.max_motion_frames < 2 or not 1 <= args.fps <= 50 or args.out.exists():
        parser.error('Use at least two motion frames, FPS in 1..50, and a new output directory')
    kind = metadata['env']
    act = submission.load_policy(kind, args.model)
    configuration = submission._CONFIGURATIONS[kind]
    for key in ('recipe', 'guard_static', 'guard_traffic'):
        if configuration.get(key, False) != metadata.get(key, False):
            parser.error(f'Checkpoint configuration differs from reference: {key}')
    expected = {(row['episode'], row['agent']): row for row in records}
    capture(args.out)
    env = submission.make_env(kind)
    world = env.unwrapped
    # RGB rendering is an offscreen view; neither reset nor step renders in this mode.
    world.render_mode = 'rgb_array'
    world.pygame_canvas.mode = 'rgb_array'
    interval = world.action_frequency
    stride = max(1, math.ceil(3000 / interval / args.max_motion_frames))
    output = dict(reference=str(args.reference), model_sha256=metadata['model_sha256'],
                  env=kind, seed=2026, recipe=metadata['recipe'], official_protocol=False,
                  recording_purpose='Development preview; selected scenarios are not a population estimate',
                  frame_stride_decisions=stride, decision_interval_seconds=interval, fps=args.fps,
                  playback_speedup=interval * stride * args.fps, final_card_hold_ms=2000,
                  episodes=[])
    selected = set(args.episodes)
    try:
        import bluesky as bs
        for episode in range(max(selected) + 1):
            obs, _ = env.reset(seed=2026 if episode == 0 else None)
            if episode not in selected:
                if kind == 'sa':
                    bs.stack.process()
                continue
            rng_before = repr(world._np_random.bit_generator.state)
            frames, final_records, steps = [], [], 0
            while True:
                if steps % stride == 0:
                    frames.append(annotated_frame(env, episode).convert('P', palette=Image.Palette.ADAPTIVE, colors=128))
                if kind == 'ma':
                    ids = list(env.agents)
                    obs, _, terms, truncs, infos = env.step({agent: act(obs[agent]) for agent in ids})
                    final_records.extend(dict(episode=episode, agent=agent, **infos[agent]) for agent in ids if terms[agent] or truncs[agent])
                    done = not env.agents
                else:
                    obs, _, terminated, truncated, info = env.step(act(obs))
                    done = terminated or truncated
                    if done:
                        final_records.append(dict(episode=episode, agent=world.agent, **info))
                steps += 1
                if done:
                    break
            final_map = {(row['episode'], row['agent']): row for row in final_records}
            expected_keys = {key for key in expected if key[0] == episode}
            if set(final_map) != expected_keys:
                raise AssertionError('Recorded aircraft identities differ from the reference')
            differences = {key: max(abs(final_map[identity][key] - expected[identity][key]) for identity in expected_keys) for key in METRICS}
            if any(value > (1e-5 if key == 'total_reward' else 0) for key, value in differences.items()):
                raise AssertionError(f'Rendered rollout differs from reference: {differences}')
            if repr(world._np_random.bit_generator.state) != rng_before:
                raise AssertionError('Recording changed the scenario RNG stream')
            # The SA renderer retains its controlled aircraft at termination; MA draws its remaining population.
            final_frame = annotated_frame(env, episode, final=True)
            final_frame.save(args.out / f'episode-{episode:03d}-final.png')
            frames.append(final_frame.convert('P', palette=Image.Palette.ADAPTIVE, colors=128))
            gif = args.out / f'episode-{episode:03d}.gif'
            frames[0].save(gif, save_all=True, append_images=frames[1:], duration=[round(1000/args.fps)]*(len(frames)-1)+[2000], loop=0, optimize=False)
            item = dict(episode=episode, file=gif.name, frames=len(frames), decisions=steps,
                        matches_reference=True, maximum_absolute_metric_differences=differences,
                        scenario_rng_unchanged=True, final_records=final_records,
                        gif_sha256=hashlib.sha256(gif.read_bytes()).hexdigest())
            output['episodes'].append(item)
            (args.out / 'recording.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
            print(json.dumps({key:value for key,value in item.items() if key != 'final_records'}), flush=True)
            frames.clear()
    finally:
        env.close()


if __name__ == '__main__':
    main()
