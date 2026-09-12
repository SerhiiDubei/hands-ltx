"""Build hands/jobs.json (180 jobs x 24 parameters) from refs.json + src/variations_*.json.

Usage:  py -3 tools/build_jobs.py            (run from the hands/ folder or anywhere)
"""
import glob
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# --- slot defaults: the ten variation types shared by every gesture ---------------------------
SLOTS = {
    "V01": dict(variation_name="breath_hold",     motion_type="micro_hold",     camera_motion="static",   motion_intensity=0.15, duration_s=5, end_state="lock_to_first", img_compression=12, first_frame_strength=1.0),
    "V02": dict(variation_name="perform_release", motion_type="gesture_cycle",  camera_motion="static",   motion_intensity=0.60, duration_s=6, end_state="lock_to_first", img_compression=12, first_frame_strength=1.0),
    "V03": dict(variation_name="exit_frame",      motion_type="exit",           camera_motion="static",   motion_intensity=0.50, duration_s=5, end_state="free",          img_compression=12, first_frame_strength=1.0),
    "V04": dict(variation_name="grain_dissolve",  motion_type="fx_material",    camera_motion="static",   motion_intensity=0.70, duration_s=6, end_state="lock_to_first", img_compression=20, first_frame_strength=0.9),
    "V05": dict(variation_name="dolly_in",        motion_type="camera",         camera_motion="dolly_in", motion_intensity=0.30, duration_s=6, end_state="free",          img_compression=12, first_frame_strength=1.0),
    "V06": dict(variation_name="orbit",           motion_type="camera",         camera_motion="orbit",    motion_intensity=0.40, duration_s=6, end_state="free",          img_compression=12, first_frame_strength=1.0),
    "V07": dict(variation_name="light_sweep",     motion_type="lighting",       camera_motion="static",   motion_intensity=0.30, duration_s=5, end_state="lock_to_first", img_compression=12, first_frame_strength=1.0),
    "V08": dict(variation_name="second_beat",     motion_type="action",         camera_motion="static",   motion_intensity=0.80, duration_s=8, end_state="free",          img_compression=12, first_frame_strength=1.0),
    "V09": dict(variation_name="interaction",     motion_type="gesture_change", camera_motion="static",   motion_intensity=0.60, duration_s=6, end_state="lock_to_first", img_compression=12, first_frame_strength=1.0),
    "V10": dict(variation_name="open_center",     motion_type="layout",         camera_motion="static",   motion_intensity=0.50, duration_s=6, end_state="free",          img_compression=12, first_frame_strength=1.0),
}

GLOBAL = dict(fps=24, sigmas_preset="distilled8", video_cfg=1.0, audio_cfg=1.0)
SEED_BASE = 20260912000

KEYS = [
    "job_id", "ref_id", "gesture", "ref_image", "variation", "variation_name",
    "motion_type", "camera_motion", "motion_intensity", "end_state",
    "prompt", "audio_prompt", "negative_prompt",
    "width", "height", "duration_s", "fps", "seed", "sigmas_preset",
    "video_cfg", "audio_cfg", "img_compression", "first_frame_strength", "output_prefix",
]


def load_sources():
    src = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "src", "variations_*.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for ref_id, variations in data.items():
            if ref_id in src:
                raise SystemExit(f"duplicate ref {ref_id} in {path}")
            src[ref_id] = variations
    return src


def main():
    with open(os.path.join(ROOT, "refs.json"), encoding="utf-8") as f:
        refs = json.load(f)
    style = refs["style"]
    sources = load_sources()

    jobs, problems = [], []
    for ref in refs["refs"]:
        rid = ref["id"]
        variations = sources.get(rid)
        if not variations:
            problems.append(f"ref {rid}: no variations source")
            continue
        seen = set()
        for v in variations:
            slot = v["slot"]
            if slot not in SLOTS:
                problems.append(f"ref {rid}: unknown slot {slot}")
                continue
            if slot in seen:
                problems.append(f"ref {rid}: duplicate slot {slot}")
            seen.add(slot)
            p = dict(SLOTS[slot])
            p.update(GLOBAL)
            p.update(v.get("overrides", {}))
            w, h = ref["gen_size"]
            w, h = p.get("width", w), p.get("height", h)
            action = v["action"].strip()
            audio = v.get("audio", "").strip()
            job = {
                "job_id": f"H{rid}_{slot}_{ref['gesture']}_{p['variation_name']}",
                "ref_id": rid,
                "gesture": ref["gesture"],
                "ref_image": "hands/" + os.path.basename(ref["file"]),
                "variation": slot,
                "variation_name": p["variation_name"],
                "motion_type": p["motion_type"],
                "camera_motion": p["camera_motion"],
                "motion_intensity": p["motion_intensity"],
                "end_state": p["end_state"],
                "prompt": f"{ref['prompt_anchor']} {action} {style['style_lock']}",
                "audio_prompt": audio,
                "negative_prompt": p.get("negative_prompt", style["negative_prompt"]),
                "width": w,
                "height": h,
                "duration_s": p["duration_s"],
                "fps": p["fps"],
                "seed": SEED_BASE + int(rid) * 100 + int(slot[1:]),
                "sigmas_preset": p["sigmas_preset"],
                "video_cfg": p["video_cfg"],
                "audio_cfg": p["audio_cfg"],
                "img_compression": p["img_compression"],
                "first_frame_strength": p["first_frame_strength"],
                "output_prefix": f"hands/{rid}_{ref['gesture']}/H{rid}_{slot}_{p['variation_name']}",
            }
            # validation
            frames = job["fps"] * job["duration_s"] + 1
            if (frames - 1) % 8:
                problems.append(f"{job['job_id']}: frames {frames} is not 8k+1")
            if job["width"] % 32 or job["height"] % 32:
                problems.append(f"{job['job_id']}: size {job['width']}x{job['height']} not multiple of 32")
            if list(job.keys()) != KEYS or len(job) != 24:
                problems.append(f"{job['job_id']}: key set mismatch")
            words = len(job["prompt"].split())
            if not (40 <= words <= 160):
                problems.append(f"{job['job_id']}: prompt has {words} words")
            jobs.append(job)
        missing = sorted(set(SLOTS) - seen)
        if missing:
            problems.append(f"ref {rid}: missing slots {missing}")

    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  -", p)
    out = os.path.join(ROOT, "jobs.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=2)

    # human-readable index for review
    total_s = sum(j["duration_s"] for j in jobs)
    lines = ["# Індекс джобів", "", f"Усього {len(jobs)} роликів, {total_s} с відео, {total_s/60:.1f} хв.", "",
             "| job_id | res | dur | end | audio | дія (початок) |", "|---|---|---|---|---|---|"]
    for j in jobs:
        action = j["prompt"].split(". ", 1)[1] if ". " in j["prompt"] else j["prompt"]
        action = action[:90].replace("|", "/")
        lines.append(f"| {j['job_id']} | {j['width']}x{j['height']} | {j['duration_s']}s | {j['end_state']} | {'yes' if j['audio_prompt'] else 'no'} | {action}… |")
    with open(os.path.join(ROOT, "jobs_index.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"jobs: {len(jobs)}  refs: {len(sources)}  problems: {len(problems)}  total video: {total_s}s")
    print("wrote", out)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
