"""Submit hands/jobs.json to ComfyUI as LTX-2.5 image-to-video jobs.

Reproduces the proven flat graph from the server history (LoadImage -> ImageScale -> LTXVPreprocess ->
LTXVAddGuide x1/x2 -> SamplerCustomAdvanced -> LTXVCropGuides -> VAEDecodeTiled -> CreateVideo -> SaveVideo)
and adds LTXVAudioVAEDecode when the job has an audio_prompt.

Usage examples (run with py -3):
  submit_jobs.py --list                              list jobs and their log status
  submit_jobs.py --filter H09_V08 --dry-run          print the graph that would be queued
  submit_jobs.py --filter H09_V08 --submit --wait    queue one job and wait for it
  submit_jobs.py --submit --limit 20                 queue the next 20 unfinished jobs, do not wait
  submit_jobs.py --status                            queue length + running job on the server
"""
import argparse
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SERVER = os.environ.get("COMFY_SERVER", "http://100.76.98.70:8188")
CLIENT_ID = "hands-batch"
LOG = os.path.join(ROOT, "jobs_log.jsonl")

MODELS = dict(
    unet="ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
    clip="gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    video_vae="ltx-2.5-video-vae-bf16.safetensors",
    audio_vae="ltx-2.5-audio-vae-bf16.safetensors",
)
SIGMAS = {
    "distilled8": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0",
}


# ----------------------------------------------------------------------------- http helpers
def get_json(path, timeout=30):
    with urllib.request.urlopen(SERVER + path, timeout=timeout) as r:
        return json.load(r)


def post_json(path, payload, timeout=60):
    req = urllib.request.Request(SERVER + path, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        raise SystemExit(f"HTTP {e.code} on {path}: {body[:2000]}")


def upload_image(local_path, subfolder="hands"):
    boundary = "----hands" + uuid.uuid4().hex
    name = os.path.basename(local_path)
    with open(local_path, "rb") as f:
        data = f.read()

    def field(n, v):
        return (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{n}\"\r\n\r\n{v}\r\n").encode()

    body = field("subfolder", subfolder) + field("overwrite", "true")
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
             f"Content-Type: image/jpeg\r\n\r\n").encode() + data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(SERVER + "/upload/image", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


# ----------------------------------------------------------------------------- graph
def build_graph(job):
    frames = job["fps"] * job["duration_s"] + 1
    lock = job["end_state"] == "lock_to_first"
    text = job["prompt"] + (" " + job["audio_prompt"] if job["audio_prompt"] else "")
    fps = float(job["fps"])
    n = {}
    n["1"] = {"class_type": "LoadImage", "inputs": {"image": job["ref_image"]}}
    n["2"] = {"class_type": "ImageScale", "inputs": {"image": ["1", 0], "upscale_method": "lanczos",
                                                     "width": job["width"], "height": job["height"], "crop": "disabled"}}
    n["3"] = {"class_type": "LTXVPreprocess", "inputs": {"image": ["2", 0], "img_compression": job["img_compression"]}}
    n["4"] = {"class_type": "UNETLoader", "inputs": {"unet_name": MODELS["unet"], "weight_dtype": "default"}}
    n["5"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": MODELS["clip"], "type": "ltxv", "device": "default"}}
    n["6"] = {"class_type": "VAELoader", "inputs": {"vae_name": MODELS["video_vae"]}}
    n["7"] = {"class_type": "VAELoader", "inputs": {"vae_name": MODELS["audio_vae"]}}
    n["8"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": text}}
    n["9"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": job["negative_prompt"]}}
    n["10"] = {"class_type": "LTXVConditioning", "inputs": {"positive": ["8", 0], "negative": ["9", 0], "frame_rate": fps}}
    n["11"] = {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": job["width"], "height": job["height"],
                                                               "length": frames, "batch_size": 1}}
    n["12"] = {"class_type": "LTXVAddGuide", "inputs": {"positive": ["10", 0], "negative": ["10", 1], "vae": ["6", 0],
                                                        "latent": ["11", 0], "image": ["3", 0], "frame_idx": 0,
                                                        "strength": job["first_frame_strength"]}}
    last = "12"
    if lock:
        n["13"] = {"class_type": "LTXVAddGuide", "inputs": {"positive": ["12", 0], "negative": ["12", 1], "vae": ["6", 0],
                                                            "latent": ["12", 2], "image": ["3", 0], "frame_idx": -1,
                                                            "strength": job["first_frame_strength"]}}
        last = "13"
    n["14"] = {"class_type": "LTXVEmptyLatentAudio", "inputs": {"frames_number": frames, "frame_rate": job["fps"],
                                                                "batch_size": 1, "audio_vae": ["7", 0]}}
    n["15"] = {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": [last, 2], "audio_latent": ["14", 0]}}
    n["16"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": job["seed"]}}
    n["17"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": SIGMAS[job["sigmas_preset"]]}}
    n["18"] = {"class_type": "SamplerEulerAncestral", "inputs": {"eta": 0.0, "s_noise": 1.0}}
    n["19"] = {"class_type": "LTXVDualCFGGuider", "inputs": {"model": ["4", 0], "positive": [last, 0], "negative": [last, 1],
                                                             "video_cfg": job["video_cfg"], "audio_cfg": job["audio_cfg"]}}
    n["20"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["16", 0], "guider": ["19", 0], "sampler": ["18", 0],
                                                                 "sigmas": ["17", 0], "latent_image": ["15", 0]}}
    n["21"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["20", 1]}}
    n["22"] = {"class_type": "LTXVCropGuides", "inputs": {"positive": [last, 0], "negative": [last, 1], "latent": ["21", 0]}}
    n["23"] = {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["22", 2], "vae": ["6", 0], "tile_size": 512,
                                                          "overlap": 64, "temporal_size": 32, "temporal_overlap": 8}}
    n["24"] = {"class_type": "CreateVideo", "inputs": {"images": ["23", 0], "fps": fps}}
    if job["audio_prompt"]:
        n["26"] = {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["21", 1], "audio_vae": ["7", 0]}}
        n["24"]["inputs"]["audio"] = ["26", 0]
    n["25"] = {"class_type": "SaveVideo", "inputs": {"video": ["24", 0], "filename_prefix": job["output_prefix"],
                                                     "format": "mp4", "format.codec": "h264",
                                                     "format.codec.encoding": "re-encode",
                                                     "format.codec.encoding.crf": 18.0}}
    return n


# ----------------------------------------------------------------------------- log
def read_log():
    done = {}
    if os.path.exists(LOG):
        with open(LOG, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                done[rec["job_id"]] = rec
    return done


def append_log(rec):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def wait_for(prompt_id, poll=5, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        hist = get_json(f"/history/{prompt_id}")
        if prompt_id in hist:
            item = hist[prompt_id]
            st = item.get("status", {})
            outs = []
            for node in item.get("outputs", {}).values():
                for v in node.values():
                    if isinstance(v, list):
                        outs += [f"{x.get('subfolder', '')}/{x.get('filename')}" for x in v if isinstance(x, dict) and "filename" in x]
            return st.get("status_str"), outs, time.time() - t0, st
        time.sleep(poll)
    return "timeout", [], time.time() - t0, {}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default=os.path.join(ROOT, "jobs.json"))
    ap.add_argument("--filter", default="", help="substring of job_id")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--wait", action="store_true", help="wait for each job before queueing the next")
    ap.add_argument("--force", action="store_true", help="resubmit jobs already logged as success")
    args = ap.parse_args()

    if args.status:
        q = get_json("/queue")
        print(f"running: {len(q['queue_running'])}  pending: {len(q['queue_pending'])}")
        for item in q["queue_running"]:
            print("  running prompt", item[1])
        return

    with open(args.jobs, encoding="utf-8") as f:
        jobs = json.load(f)
    done = read_log()
    sel = [j for j in jobs if args.filter in j["job_id"]]
    if not args.force:
        sel = [j for j in sel if done.get(j["job_id"], {}).get("status") != "success"]
    if args.limit:
        sel = sel[: args.limit]

    if args.list:
        for j in sel:
            st = done.get(j["job_id"], {}).get("status", "-")
            print(f"{st:>8}  {j['job_id']}  {j['width']}x{j['height']}  {j['duration_s']}s  {j['end_state']}")
        print(f"{len(sel)} jobs selected")
        return

    if args.dry_run:
        for j in sel[:1]:
            print(json.dumps(build_graph(j), ensure_ascii=False, indent=1))
        print(f"{len(sel)} jobs would be queued")
        return

    if not args.submit:
        ap.print_help()
        return

    uploaded = set()
    for j in sel:
        if j["ref_image"] not in uploaded:
            local = os.path.join(ROOT, "ref", os.path.basename(j["ref_image"]))
            res = upload_image(local)
            print("uploaded", res)
            uploaded.add(j["ref_image"])
        t_sub = time.time()
        res = post_json("/prompt", {"prompt": build_graph(j), "client_id": CLIENT_ID})
        pid = res.get("prompt_id")
        errs = res.get("node_errors") or {}
        print(f"queued {j['job_id']} -> {pid}" + (f"  node_errors={errs}" if errs else ""))
        rec = {"job_id": j["job_id"], "prompt_id": pid, "status": "queued", "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "seed": j["seed"], "size": f"{j['width']}x{j['height']}", "duration_s": j["duration_s"]}
        if args.wait:
            status, outs, secs, st = wait_for(pid)
            rec.update(status=status, seconds=round(secs, 1), outputs=outs)
            if status != "success":
                rec["messages"] = st.get("messages", [])[-3:]
            print(f"  {status} in {secs:.0f}s  outputs={outs}")
        append_log(rec)


if __name__ == "__main__":
    main()
