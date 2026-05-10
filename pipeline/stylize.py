"""Stage 3: stylize keyframes through ComfyUI.

We process keyframes **in order** so each one can be conditioned on the
previous stylized output (latent blend) — this is what cuts texture
boiling between sparse keyframes.

Communication with ComfyUI is over its built-in HTTP + WebSocket API:

  * ``POST /upload/image``  - send the input frame
  * ``POST /prompt``        - submit the workflow graph (returns prompt_id)
  * ``WS  /ws``             - block until ``executing`` reports done
  * ``GET /history/<id>``   - fetch output image filenames
  * ``GET /view``           - download the produced image
"""

from __future__ import annotations

import io
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

import requests
import websocket  # websocket-client
from PIL import Image
from tqdm import tqdm

from .config import PipelineConfig
from .keyframes import load_manifest
from .utils import get_logger, frame_path


# ---------------------------------------------------------------------------
# Workflow templating
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"__[A-Z][A-Z0-9_]*__")


def _drop_optional_prev(graph: dict) -> dict:
    """Remove nodes flagged ``_optional_prev`` and rewire KSampler -> direct latent.

    Specific to the SDXL workflow template, which marks the prev-frame branch
    with ``_optional_prev: true`` so it can be stripped on the very first
    keyframe (when there is no previous stylized output yet).
    """
    g = {k: v for k, v in graph.items()
         if isinstance(v, dict) and not v.get("_optional_prev")}
    # KSampler should pull straight from VAEEncode of current frame (node 30)
    if "40" in g:
        g["40"]["inputs"]["latent_image"] = ["30", 0]
    return g


def _materialize_workflow(cfg: PipelineConfig, *,
                          input_image: str,
                          prev_stylized: str | None,
                          keyframe_anchor: str | None,
                          out_prefix: str) -> dict:
    raw = Path(cfg.comfy_workflow).read_text()
    template_tokens = set(_PLACEHOLDER_RE.findall(raw))
    has_prev = prev_stylized is not None
    prev_ref = prev_stylized or input_image
    anchor_ref = keyframe_anchor or input_image

    init_latent_node = "32" if has_prev else "30"

    repl = {
        "__CKPT__": cfg.comfy_ckpt,
        "__CONTROLNET__": cfg.comfy_controlnet,
        "__PROMPT__": cfg.prompt,
        "__NEG_PROMPT__": cfg.negative_prompt,
        "__INPUT_IMAGE__": input_image,
        "__PREV_STYLIZED__": prev_ref,
        "__KEYFRAME_ANCHOR__": anchor_ref,
        "__OUT_PREFIX__": out_prefix,
        "__INIT_LATENT_NODE__": init_latent_node,
        "__SAMPLER__": cfg.sampler,
        "__SCHEDULER__": cfg.scheduler,
    }
    numeric = {
        "__CANNY_LOW__": cfg.canny_low,
        "__CANNY_HIGH__": cfg.canny_high,
        "__CN_STRENGTH__": cfg.controlnet_strength,
        "__PREV_BLEND__": cfg.prev_frame_blend,
        "__SEED__": cfg.seed,
        "__STEPS__": cfg.steps,
        "__CFG__": cfg.cfg,
        "__DENOISE__": cfg.denoise,
    }

    # String substitutions stay quoted in the JSON.
    for key, val in repl.items():
        raw = raw.replace(key, str(val))
    # Numeric substitutions: the template wraps them in quotes so it stays
    # valid JSON; we replace the *whole* "__TOKEN__" with the raw literal.
    for key, val in numeric.items():
        raw = raw.replace(f'"{key}"', json.dumps(val))

    graph = json.loads(raw)
    graph.pop("_comment", None)
    if not has_prev:
        graph = _drop_optional_prev(graph)
    # _optional_prev is not a real Comfy field - strip it.
    for node in graph.values():
        if isinstance(node, dict):
            node.pop("_optional_prev", None)

    known = set(repl) | set(numeric)
    unresolved = sorted(t for t in template_tokens if t not in known)
    if unresolved:
        raise RuntimeError(
            f"workflow {cfg.comfy_workflow} contains unknown placeholders: "
            f"{', '.join(unresolved)}. Either remove them from the template or "
            f"add a binding in stylize._materialize_workflow."
        )
    return graph


# ---------------------------------------------------------------------------
# ComfyUI client
# ---------------------------------------------------------------------------

class ComfyClient:
    def __init__(self, host: str):
        self.host = host
        self.client_id = str(uuid.uuid4())

    def upload_image(self, path: Path, *, name: str | None = None) -> str:
        name = name or path.name
        with open(path, "rb") as fh:
            r = requests.post(
                f"http://{self.host}/upload/image",
                files={"image": (name, fh, "image/png")},
                data={"overwrite": "true"},
                timeout=60,
            )
        if not r.ok:
            raise RuntimeError(
                f"ComfyUI /upload/image {r.status_code}: {r.text[:2000]}"
            )
        return r.json()["name"]

    def queue_prompt(self, graph: dict) -> str:
        r = requests.post(
            f"http://{self.host}/prompt",
            json={"prompt": graph, "client_id": self.client_id},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["prompt_id"]

    def wait(self, prompt_id: str, *, timeout: float = 600.0) -> None:
        ws = websocket.WebSocket()
        ws.connect(f"ws://{self.host}/ws?clientId={self.client_id}",
                   timeout=timeout)
        deadline = time.time() + timeout
        try:
            while time.time() < deadline:
                msg = ws.recv()
                if not isinstance(msg, str):
                    continue
                data = json.loads(msg)
                if data.get("type") != "executing":
                    continue
                d = data["data"]
                if d.get("prompt_id") == prompt_id and d.get("node") is None:
                    return
            raise TimeoutError(f"prompt {prompt_id} did not finish in {timeout}s")
        finally:
            ws.close()

    def fetch_output(self, prompt_id: str, save_to: Path) -> Path:
        h = requests.get(f"http://{self.host}/history/{prompt_id}",
                         timeout=30).json()[prompt_id]
        for node_out in h["outputs"].values():
            for img in node_out.get("images", []):
                params = {"filename": img["filename"],
                          "subfolder": img.get("subfolder", ""),
                          "type": img.get("type", "output")}
                r = requests.get(f"http://{self.host}/view",
                                 params=params, timeout=60)
                r.raise_for_status()
                Image.open(io.BytesIO(r.content)).save(save_to)
                return save_to
        raise RuntimeError(f"no images in history for {prompt_id}")


# ---------------------------------------------------------------------------
# Stage entry
# ---------------------------------------------------------------------------

def run_stage(cfg: PipelineConfig) -> int:
    """Stylize every keyframe sequentially. Returns the number processed."""
    log = get_logger("stylize")
    manifest = load_manifest(cfg)
    kf_dir = Path(cfg.paths["keyframes"])
    out_dir = Path(cfg.paths["stylized"])
    for old in out_dir.glob("*.png"):
        old.unlink()

    client = ComfyClient(cfg.comfy_host)
    remote_by_ordinal: dict[int, str] = {}

    for entry in tqdm(manifest["keyframes"], desc="stylize"):
        ord_ = entry["keyframe_ordinal"]
        prev_ord = entry.get("prev_keyframe_ordinal")
        anchor_ord = entry.get("anchor_keyframe_ordinal")
        local_in = frame_path(kf_dir, ord_)
        local_out = frame_path(out_dir, ord_)

        remote_in = client.upload_image(local_in,
                                        name=f"kf_{ord_:06d}.png")
        prev_remote = (
            remote_by_ordinal.get(prev_ord)
            if cfg.use_prev_reference and prev_ord is not None
            else None
        )
        anchor_remote = (
            remote_by_ordinal.get(anchor_ord)
            if cfg.use_anchor_reference and anchor_ord is not None
            else None
        )
        graph = _materialize_workflow(
            cfg,
            input_image=remote_in,
            prev_stylized=prev_remote,
            keyframe_anchor=anchor_remote,
            out_prefix=f"styl_{ord_:06d}",
        )
        pid = client.queue_prompt(graph)
        client.wait(pid)
        client.fetch_output(pid, local_out)

        # The freshly produced image must be re-uploaded as input for the next
        # iteration so its name is reachable inside ComfyUI's input dir.
        remote_by_ordinal[ord_] = client.upload_image(local_out,
                                                      name=f"prev_{ord_:06d}.png")

    log.info("stylized %d keyframes -> %s",
             len(manifest["keyframes"]), out_dir)
    return len(manifest["keyframes"])
