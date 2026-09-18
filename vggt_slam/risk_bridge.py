"""TCP JSON-lines bridge so mechlmm_risk can consume VGGT-SLAM without importing it.

Default bind: 127.0.0.1:8765. Disable with --risk_bridge_port 0.

Messages (one JSON object per line):

  server → client
    {"type": "pose", "timestamp_s", "keyframe_id", "submap_id",
     "tracking_ok", "T_world_cam" (4x4), "loop_closed"}
    {"type": "query_result", "id", "query", "hits": [...], "error": null}

  client → server
    {"type": "get_pose"}
    {"type": "query", "id": "...", "query": "kettle"}
    {"type": "ping"}
"""

from __future__ import annotations

import json
import socket
import threading
import time
import traceback
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import numpy as np

IDENTITY_4X4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return super().default(obj)


def _json_line(payload: dict) -> bytes:
    return (json.dumps(payload, cls=_NumpyEncoder) + "\n").encode("utf-8")


def _keyframe_id(submap_id, frame_index: int) -> str:
    return f"{submap_id}/{frame_index}"


def latest_pose_payload(solver, loop_closed: bool = False) -> dict:
    """Camera-to-world pose of the latest non-loop keyframe, or tracking_ok=false."""
    now = time.time()
    try:
        if solver.map.get_num_submaps() == 0:
            return {
                "type": "pose",
                "timestamp_s": now,
                "keyframe_id": "",
                "submap_id": "",
                "tracking_ok": False,
                "T_world_cam": IDENTITY_4X4,
                "loop_closed": False,
            }
        submap = solver.map.get_latest_submap(ignore_loop_closure_submaps=True)
        poses = submap.get_all_poses_world(solver.graph)
        frame_index = int(submap.get_last_non_loop_frame_index() or (len(poses) - 1))
        frame_index = max(0, min(frame_index, len(poses) - 1))
        pose = poses[frame_index]
        submap_id = str(submap.get_id())
        return {
            "type": "pose",
            "timestamp_s": now,
            "keyframe_id": _keyframe_id(submap_id, frame_index),
            "submap_id": submap_id,
            "tracking_ok": True,
            "T_world_cam": np.asarray(pose, dtype=float).tolist(),
            "loop_closed": bool(loop_closed),
        }
    except Exception as exc:
        return {
            "type": "pose",
            "timestamp_s": now,
            "keyframe_id": "",
            "submap_id": "",
            "tracking_ok": False,
            "T_world_cam": IDENTITY_4X4,
            "loop_closed": bool(loop_closed),
            "error": str(exc),
        }


DEFAULT_SCAN_PROMPTS = [
    "television",
    "laptop",
    "computer monitor",
    "phone",
    "kettle",
    "stove",
    "oven",
    "microwave",
    "knife",
    "scissors",
    "mug",
    "cup",
    "bottle",
    "glass",
    "person",
    "chair",
    "table",
    "sofa",
    "lamp",
    "sink",
    "refrigerator",
]

PROMPT_EXPAND = {
    "tv": ["television", "tv", "television screen"],
    "television": ["television", "television screen"],
    "laptop": ["laptop", "laptop computer"],
    "phone": ["phone", "mobile phone"],
    "person": ["person", "human"],
}

LABEL_ALIASES = {
    "tv": "television",
    "tele": "television",
    "tv screen": "television",
    "flatscreen": "television",
    "computer": "laptop",
    "fridge": "refrigerator",
    "couch": "sofa",
}


def _canonical_label(label: str) -> str:
    needle = " ".join((label or "").lower().split())
    return LABEL_ALIASES.get(needle, needle)


def _expand_prompts(query: str) -> List[str]:
    key = _canonical_label(query)
    extra = PROMPT_EXPAND.get(query.lower().strip()) or PROMPT_EXPAND.get(key) or []
    ordered = []
    for prompt in [query.strip(), key, *extra]:
        if prompt and prompt not in ordered:
            ordered.append(prompt)
    return ordered


def _best_mask_hit(found_submap, graph, frame_index, keyframe_id, submap_id, query, label, masks, scores, clip_score=0.0):
    import vggt_slam.slam_utils as utils

    n = int(masks.shape[0]) if masks is not None else 0
    if n == 0:
        return None
    score_list = scores.detach().float().cpu().numpy().reshape(-1) if scores is not None else []
    best_i = int(np.argmax(score_list)) if len(score_list) else 0
    mask = masks[best_i].detach().float().cpu().numpy()
    confidence = float(score_list[best_i]) if best_i < len(score_list) else 0.0
    try:
        obb_center, obb_extent, obb_rotation = utils.compute_obb_from_points(
            found_submap.get_points_in_mask(frame_index, mask, graph)
        )
    except Exception as exc:
        print(f"[risk_bridge] OBB failed for {query!r} ({exc}); using camera xyz")
        try:
            poses = found_submap.get_all_poses_world(graph)
            idx = max(0, min(int(frame_index), len(poses) - 1))
            obb_center = np.asarray(poses[idx], dtype=float)[:3, 3]
        except Exception:
            obb_center = np.zeros(3)
        obb_extent = np.array([0.3, 0.3, 0.3], dtype=float)
        obb_rotation = np.eye(3)
    return {
        "query": query,
        "label": label,
        "keyframe_id": keyframe_id,
        "submap_id": str(submap_id),
        "confidence": confidence,
        "clip_score": float(clip_score),
        "obb_center": np.asarray(obb_center, dtype=float).tolist(),
        "obb_extents": np.asarray(obb_extent, dtype=float).tolist(),
        "obb_rotation": np.asarray(obb_rotation, dtype=float).tolist(),
    }


def _latest_frame(solver):
    submap = solver.map.get_latest_submap(ignore_loop_closure_submaps=True)
    frame_index = int(submap.get_last_non_loop_frame_index() or 0)
    frame_index = max(0, frame_index)
    return submap, submap.get_id(), frame_index


def _candidate_frame_indices(submap) -> List[int]:
    last = int(submap.get_last_non_loop_frame_index() or 0)
    frames = [last, 0, last // 2]
    out = []
    for idx in frames:
        if idx not in out and idx >= 0:
            out.append(idx)
    return out


def run_object_query(solver, clip_model, clip_tokenizer, processor, query: str) -> dict:
    """Text → SAM3 mask → 3D OBB. Tries aliases. Without CLIP, searches the latest submap."""
    from torchvision.transforms.functional import to_pil_image
    import torch
    import vggt_slam.slam_utils as utils

    if solver.map.get_num_submaps() == 0:
        return {"hits": [], "error": "no_map"}
    if processor is None:
        return {"hits": [], "error": "sam3_not_loaded"}

    prompts = _expand_prompts(query)
    label = _canonical_label(query)
    hits: List[Dict[str, Any]] = []

    if clip_model is not None and clip_tokenizer is not None:
        text_emb = utils.compute_text_embeddings(clip_model, clip_tokenizer, prompts[0])
        clip_score, submap_id, frame_index = solver.map.retrieve_best_semantic_frame(text_emb)
        found_submap = solver.map.get_submap(submap_id)
        keyframe_id = _keyframe_id(submap_id, frame_index)
        with torch.no_grad():
            pil_img = to_pil_image(found_submap.get_frame_at_index(frame_index))
            state = processor.set_image(pil_img)
            for prompt in prompts:
                output = processor.set_text_prompt(state=state, prompt=prompt)
                hit = _best_mask_hit(
                    found_submap, solver.graph, frame_index, keyframe_id, submap_id,
                    query, label, output.get("masks"), output.get("scores"), clip_score,
                )
                if hit:
                    hits.append(hit)
                    break
        return {"hits": hits, "error": None}

    submap = solver.map.get_latest_submap(ignore_loop_closure_submaps=True)
    submap_id = submap.get_id()
    with torch.no_grad():
        for frame_index in _candidate_frame_indices(submap):
            pil_img = to_pil_image(submap.get_frame_at_index(frame_index))
            state = processor.set_image(pil_img)
            keyframe_id = _keyframe_id(submap_id, frame_index)
            for prompt in prompts:
                output = processor.set_text_prompt(state=state, prompt=prompt)
                hit = _best_mask_hit(
                    submap, solver.graph, frame_index, keyframe_id, submap_id,
                    query, label, output.get("masks"), output.get("scores"), 0.0,
                )
                if hit:
                    hits.append(hit)
                    return {"hits": hits, "error": None}
    return {"hits": hits, "error": None}


def scan_latest_keyframe(solver, processor, prompts: List[str], sam3_lock=None, data_lock=None) -> dict:
    """One set_image, then each watch prompt on the latest keyframe. Best instance per label.

    SAM3 runs under sam3_lock only. Map reads/OBB use data_lock so SLAM can keep adding submaps.
    """
    from torchvision.transforms.functional import to_pil_image
    import torch

    map_lock = data_lock if data_lock is not None else nullcontext()
    lock = sam3_lock if sam3_lock is not None else threading.Lock()

    with map_lock:
        if solver.map.get_num_submaps() == 0:
            return {"hits": [], "error": "no_map", "keyframe_id": "", "submap_id": ""}
        if processor is None:
            return {"hits": [], "error": "sam3_not_loaded", "keyframe_id": "", "submap_id": ""}
        submap, submap_id, frame_index = _latest_frame(solver)
        keyframe_id = _keyframe_id(submap_id, frame_index)
        pil_img = to_pil_image(submap.get_frame_at_index(frame_index))

    hits: List[Dict[str, Any]] = []
    seen_labels = set()
    pending: List[tuple] = []
    mask_counts: Dict[str, int] = {}

    print(f"[risk_bridge] scanning {len(prompts)} prompts on keyframe {keyframe_id} ...")
    old_thresh = getattr(processor, "confidence_threshold", None)
    if old_thresh is not None:
        try:
            processor.confidence_threshold = min(float(old_thresh), 0.3)
        except Exception:
            pass
    try:
        with lock:
            with torch.no_grad():
                state = processor.set_image(pil_img)
                for prompt in prompts:
                    label = _canonical_label(prompt)
                    if label in seen_labels:
                        continue
                    try:
                        output = processor.set_text_prompt(state=state, prompt=prompt)
                    except Exception as exc:
                        print(f"[risk_bridge] SAM3 prompt {prompt!r} failed: {exc}")
                        mask_counts[prompt] = -1
                        continue
                    masks = output.get("masks")
                    scores = output.get("scores")
                    n = int(masks.shape[0]) if masks is not None else 0
                    mask_counts[prompt] = n
                    if n == 0:
                        continue
                    pending.append((prompt, label, masks, scores))
                    seen_labels.add(label)
    finally:
        if old_thresh is not None:
            try:
                processor.confidence_threshold = old_thresh
            except Exception:
                pass

    found = [k for k, n in mask_counts.items() if n > 0]
    print(f"[risk_bridge] SAM3 masks with hits: {found or 'none'}  (checked {len(mask_counts)})")

    with map_lock:
        for prompt, label, masks, scores in pending:
            hit = _best_mask_hit(
                submap, solver.graph, frame_index, keyframe_id, submap_id,
                prompt, label, masks, scores, 0.0,
            )
            if hit:
                hits.append(hit)
                print(f"[risk_bridge]   found {label} conf={hit['confidence']:.2f}")

    return {"hits": hits, "error": None, "keyframe_id": keyframe_id, "submap_id": str(submap_id)}


class RiskBridge:
    """Background TCP server. Pose ticks are pushed after each submap."""

    def __init__(
        self,
        solver,
        data_lock: threading.Lock,
        solver_lock: threading.Lock,
        clip_model=None,
        clip_tokenizer=None,
        processor=None,
        sam3_lock: Optional[threading.Lock] = None,
        host: str = "127.0.0.1",
        port: int = 8765,
    ):
        self.solver = solver
        self.data_lock = data_lock
        self.solver_lock = solver_lock
        self.sam3_lock = sam3_lock or threading.Lock()
        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer
        self.processor = processor
        self.host = host
        self.port = port
        self._clients: List[socket.socket] = []
        self._clients_lock = threading.Lock()
        self._stop = threading.Event()
        self._server_sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._latest_pose: dict = latest_pose_payload(solver, loop_closed=False)
        self._watch_prompts: List[str] = list(DEFAULT_SCAN_PROMPTS)
        self._scan_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._scan_again = False
        self._scan_running = False
        self._scan_state_lock = threading.Lock()

    def start(self) -> None:
        if self.port <= 0:
            print("[risk_bridge] disabled (port 0)")
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((self.host, self.port))
        except OSError as exc:
            print(f"[risk_bridge] could not bind {self.host}:{self.port} ({exc}); continuing without adapter")
            sock.close()
            return
        sock.listen(8)
        sock.settimeout(0.5)
        self._server_sock = sock
        self._thread = threading.Thread(target=self._accept_loop, daemon=True, name="vggt-risk-bridge")
        self._thread.start()
        print(f"[risk_bridge] listening on {self.host}:{self.port}")

    def stop(self) -> None:
        self._stop.set()
        with self._clients_lock:
            clients = list(self._clients)
            self._clients.clear()
        for client in clients:
            try:
                client.close()
            except OSError:
                pass
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass

    def publish_latest_pose(self, loop_closed: bool = False) -> None:
        with self.data_lock:
            payload = latest_pose_payload(self.solver, loop_closed=loop_closed)
        self._latest_pose = payload
        self.broadcast(payload)
        kf = payload.get("keyframe_id") or "-"
        ok = payload.get("tracking_ok")
        print(f"[risk_bridge] pose keyframe={kf} tracking_ok={ok} loop_closed={loop_closed}")
        if payload.get("tracking_ok"):
            self._schedule_scan()

    def _schedule_scan(self) -> None:
        with self._clients_lock:
            n_clients = len(self._clients)
        if n_clients == 0:
            print("[risk_bridge] auto-scan skipped (no mapper client connected)")
            return
        if not self.processor:
            print("[risk_bridge] auto-scan skipped (SAM3 not loaded — start SLAM with --mask_prompt)")
            self.broadcast({
                "type": "scan_status",
                "error": "sam3_not_loaded",
                "hits": 0,
                "keyframe_id": (self._latest_pose or {}).get("keyframe_id"),
            })
            return
        if not self._watch_prompts:
            print("[risk_bridge] auto-scan skipped (empty watch list)")
            return
        with self._scan_state_lock:
            if self._scan_running:
                self._scan_again = True
                print("[risk_bridge] auto-scan queued (one already running)")
                return
            self._scan_running = True
        threading.Thread(target=self._scan_worker, daemon=True, name="vggt-object-scan").start()

    def _scan_worker(self) -> None:
        try:
            print("[risk_bridge] auto-scan waiting for SLAM GPU...")
            result = self._run_scan()
            payload = {
                "type": "scan_result",
                "id": None,
                "query": "scan",
                "hits": result.get("hits") or [],
                "error": result.get("error"),
                "keyframe_id": result.get("keyframe_id"),
                "submap_id": result.get("submap_id"),
            }
            n = len(payload["hits"])
            labels = [h.get("label") for h in payload["hits"]]
            print(f"[risk_bridge] auto-scan kf={payload.get('keyframe_id')} → {n} object(s) {labels}")
            self.broadcast(payload)
        except Exception:
            traceback.print_exc()
            self.broadcast({
                "type": "scan_status",
                "error": "scan_exception",
                "hits": 0,
            })
        finally:
            again = False
            with self._scan_state_lock:
                if self._scan_again:
                    self._scan_again = False
                    again = True
                else:
                    self._scan_running = False
            if again:
                threading.Thread(target=self._scan_worker, daemon=True, name="vggt-object-scan").start()

    def _run_scan(self, prompts: Optional[List[str]] = None) -> dict:
        use = list(prompts) if prompts else list(self._watch_prompts)
        # Don't overlap SAM3 with VGGT on the same GPU.
        with self.solver_lock:
            return scan_latest_keyframe(
                self.solver, self.processor, use, self.sam3_lock, self.data_lock
            )

    def broadcast(self, payload: dict) -> None:
        data = _json_line(payload)
        dead = []
        with self._clients_lock:
            clients = list(self._clients)
        for client in clients:
            try:
                with self._send_lock:
                    client.sendall(data)
            except OSError:
                dead.append(client)
        if dead:
            with self._clients_lock:
                for client in dead:
                    if client in self._clients:
                        self._clients.remove(client)
                    try:
                        client.close()
                    except OSError:
                        pass

    def _accept_loop(self) -> None:
        assert self._server_sock is not None
        while not self._stop.is_set():
            try:
                client, addr = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            print(f"[risk_bridge] client {addr[0]}:{addr[1]}")
            with self._clients_lock:
                self._clients.append(client)
            threading.Thread(
                target=self._client_loop,
                args=(client,),
                daemon=True,
                name=f"vggt-risk-client-{addr[1]}",
            ).start()
            try:
                with self._send_lock:
                    client.sendall(_json_line(self._latest_pose))
            except OSError:
                pass
            if (self._latest_pose or {}).get("tracking_ok"):
                self._schedule_scan()

    def _client_loop(self, client: socket.socket) -> None:
        buffer = b""
        client.settimeout(1.0)
        try:
            while not self._stop.is_set():
                try:
                    chunk = client.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    raw, buffer = buffer.split(b"\n", 1)
                    if not raw.strip():
                        continue
                    try:
                        message = json.loads(raw.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue
                    self._handle(client, message)
        finally:
            with self._clients_lock:
                if client in self._clients:
                    self._clients.remove(client)
            try:
                client.close()
            except OSError:
                pass

    def _handle(self, client: socket.socket, message: dict) -> None:
        kind = message.get("type")
        if kind == "ping":
            self._send(client, {"type": "pong"})
            return
        if kind == "get_pose":
            with self.data_lock:
                payload = latest_pose_payload(self.solver)
            self._latest_pose = payload
            self._send(client, payload)
            return
        if kind == "query":
            query = str(message.get("query") or "").strip()
            request_id = message.get("id")
            if not query:
                self._send(
                    client,
                    {"type": "query_result", "id": request_id, "query": query, "hits": [], "error": "empty_query"},
                )
                return
            try:
                # Wait if a submap is currently running so we do not contend for the GPU.
                with self.solver_lock:
                    with self.data_lock:
                        with self.sam3_lock:
                            result = run_object_query(
                                self.solver,
                                self.clip_model,
                                self.clip_tokenizer,
                                self.processor,
                                query,
                            )
            except Exception as exc:
                traceback.print_exc()
                result = {"hits": [], "error": str(exc)}
            self._send(
                client,
                {
                    "type": "query_result",
                    "id": request_id,
                    "query": query,
                    "hits": result.get("hits") or [],
                    "error": result.get("error"),
                },
            )
            n = len(result.get("hits") or [])
            print(f"[risk_bridge] query {query!r} → {n} hit(s) error={result.get('error')}")
            return
        if kind == "watch":
            prompts = message.get("prompts") or DEFAULT_SCAN_PROMPTS
            self._watch_prompts = [str(p).strip() for p in prompts if str(p).strip()]
            print(f"[risk_bridge] watch list: {self._watch_prompts}")
            self._send(client, {"type": "watch_ok", "prompts": self._watch_prompts})
            self._schedule_scan()
            return
        if kind == "scan":
            if message.get("id"):
                threading.Thread(
                    target=self._handle_scan,
                    args=(client, message),
                    daemon=True,
                    name="vggt-risk-scan",
                ).start()
            else:
                self._schedule_scan()
            return

    def _handle_scan(self, client: socket.socket, message: dict) -> None:
        request_id = message.get("id")
        prompts = message.get("prompts")
        try:
            result = self._run_scan(prompts)
        except Exception as exc:
            traceback.print_exc()
            result = {"hits": [], "error": str(exc), "keyframe_id": "", "submap_id": ""}
        self._send(
            client,
            {
                "type": "scan_result",
                "id": request_id,
                "query": "scan",
                "hits": result.get("hits") or [],
                "error": result.get("error"),
                "keyframe_id": result.get("keyframe_id"),
                "submap_id": result.get("submap_id"),
            },
        )
        n = len(result.get("hits") or [])
        print(f"[risk_bridge] scan → {n} object(s) error={result.get('error')}")

    def _send(self, client: socket.socket, payload: dict) -> None:
        try:
            with self._send_lock:
                client.sendall(_json_line(payload))
        except OSError:
            pass
