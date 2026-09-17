#!/usr/bin/env python3
"""Persistent JSON-lines worker for the native DeSynth Web UI.

The worker deliberately uses stdin/stdout instead of HTTP. stdout is reserved
for protocol messages; diagnostics from PyTorch/Diffusers are sent to stderr.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import re
import secrets
import sys
import tempfile
import traceback
import warnings
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "out"
PYTHON = ROOT / ".venv" / "bin" / "python"
TRANSFORMER_CANDIDATES = (
    ROOT / "qwen-image-2512-Q4_K_M.gguf",
    ROOT / "Qwen-Image-2512-Q4_K_M.gguf",
)
LORA = ROOT / "Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors"
EMBEDS = ROOT / "embeds_cache.pt"

_runtime: dict[str, Any] | None = None


def emit(event_type: str, **payload: Any) -> None:
    message = {"type": event_type, **payload}
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def environment_checks() -> list[dict[str, Any]]:
    return [
        {"label": "Python 环境", "ok": PYTHON.is_file()},
        {"label": "GGUF 模型", "ok": any(path.is_file() for path in TRANSFORMER_CANDIDATES)},
        {"label": "Lightning LoRA", "ok": LORA.is_file()},
        {"label": "Prompt Embeddings", "ok": EMBEDS.is_file()},
    ]


def _as_float(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是数字")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数字") from exc
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} 必须在 {minimum:g} 到 {maximum:g} 之间")
    return result


def _as_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} 必须是布尔值")
    return value


def _as_choice(value: Any, name: str, choices: set[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{name} 选项无效")
    return value


def _as_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是整数")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是整数") from exc
    if str(value).strip() not in {str(result), f"{result}.0"}:
        raise ValueError(f"{name} 必须是整数")
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} 必须在 {minimum} 到 {maximum} 之间")
    return result


def validate_settings(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("设置格式无效")
    if raw.get("authorized") is not True:
        raise ValueError("请先确认图片用于研究、教育或已获授权的评估")

    restore = _as_bool(raw.get("restore", True), "细节恢复选项")
    restore_mode = (
        _as_choice(raw.get("restoreMode", "gaussian"), "恢复模式", {"gaussian", "edge"})
        if restore
        else "gaussian"
    )
    device = _as_choice(raw.get("device", "auto"), "设备", {"auto", "mps", "cpu"})

    seed_value = raw.get("seed")
    seed = None
    if seed_value not in (None, ""):
        seed = _as_int(seed_value, "Seed", 0, 2**63 - 1)

    return {
        "denoise": _as_float(raw.get("denoise", 0.25), "降噪强度", 0.01, 1.0),
        "steps": _as_int(raw.get("steps", 8), "采样步数", 1, 100),
        "passes": _as_int(raw.get("passes", 1), "处理轮数", 1, 8),
        "restore": restore,
        "restore_mode": restore_mode,
        "restore_sigma": (
            _as_float(raw.get("restoreSigma", 1.95), "Sigma", 0.1, 4.0) if restore else 1.95
        ),
        "unsharp": _as_float(raw.get("unsharp", 0.0), "锐化", 0.0, 3.0) if restore else 0.0,
        "keep_intermediate": _as_bool(raw.get("keepIntermediate", False), "中间图选项"),
        "device": device,
        "seed": seed,
    }


def _safe_stem(path: Path) -> str:
    stem = re.sub(r"[^\w.-]+", "_", path.stem, flags=re.UNICODE).strip("._")
    return stem[:100] or "image"


def _save_png_unique(image: Any, desired_path: Path) -> Path:
    """Publish a complete PNG without following or overwriting an existing path."""
    desired_path.parent.mkdir(exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=".desynth-webui-",
            suffix=".png",
            dir=desired_path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            image.save(handle, format="PNG")
            handle.flush()
            os.fsync(handle.fileno())

        for index in range(1, 10_000):
            candidate = desired_path if index == 1 else desired_path.with_name(
                f"{desired_path.stem}_{index}{desired_path.suffix}"
            )
            try:
                os.link(temporary_path, candidate, follow_symlinks=False)
            except FileExistsError:
                continue
            return candidate
        raise RuntimeError("out 文件夹中同名结果过多，请先整理输出文件")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _validate_input_image(input_path: Path) -> None:
    from PIL import Image

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(input_path) as candidate:
                width, height = candidate.size
                if width <= 0 or height <= 0 or width * height > 64_000_000:
                    raise ValueError("图片尺寸无效或超过 6400 万像素")
                candidate.verify()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("图片像素尺寸过大") from exc
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("无法解码输入图片，请换用有效的 PNG、JPEG、WebP 或 TIFF") from exc


def _unload_runtime() -> None:
    global _runtime
    if _runtime is None:
        return
    _runtime = None
    gc.collect()
    try:
        import torch

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _get_runtime(device: str) -> tuple[Any, Any, Any, str]:
    global _runtime

    emit("state", title="加载模型", message="正在初始化 PyTorch、Qwen-Image 与本机加速器。首次运行会花更久。")
    with contextlib.redirect_stdout(sys.stderr):
        import desynth as core

    resolved = core._resolve_device(device)
    if _runtime is not None and _runtime["device"] == resolved:
        emit("log", message=f"复用已加载的模型（{resolved}）")
        return core, _runtime["pipeline"], _runtime["embeds"], resolved

    if _runtime is not None:
        emit("log", message="设备发生变化，正在重新加载模型")
        _unload_runtime()

    emit("log", message=f"加载 GGUF 与 Lightning LoRA（{resolved}）")
    with contextlib.redirect_stdout(sys.stderr):
        pipeline = core.build_pipeline(device=resolved)
        embeds = core.torch.load(core.EMBEDS_CACHE, map_location="cpu", weights_only=True)
    try:
        pipeline.set_progress_bar_config(disable=True)
    except Exception:
        pass
    _runtime = {"device": resolved, "pipeline": pipeline, "embeds": embeds}
    emit("log", message="模型已加载，后续任务会直接复用")
    return core, pipeline, embeds, resolved


def _run(request: dict[str, Any]) -> None:
    input_value = request.get("input")
    if not isinstance(input_value, str) or not input_value:
        raise ValueError("没有选择输入图片")
    input_path = Path(input_value).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到输入图片：{input_path.name}")
    if input_path.stat().st_size > 100 * 1024 * 1024:
        raise ValueError("输入图片超过 100 MB")
    _validate_input_image(input_path)

    settings = validate_settings(request.get("settings"))
    seed = settings["seed"] if settings["seed"] is not None else secrets.randbits(63)
    emit("log", message=f"Seed: {seed}{'（固定）' if settings['seed'] is not None else '（随机）'}")

    try:
        core, pipeline, embeds, resolved_device = _get_runtime(settings["device"])
        emit("state", title="读取图片", message=f"正在读取 {input_path.name}")
        with contextlib.redirect_stdout(sys.stderr):
            image = core.load_image(str(input_path))
        emit("log", message=f"输入尺寸：{image.size[0]} × {image.size[1]} · 设备：{resolved_device}")

        current = image
        for pass_index in range(settings["passes"]):
            emit(
                "state",
                title="图像推理",
                message=f"正在执行第 {pass_index + 1} / {settings['passes']} 轮 img2img 推理。",
            )
            with contextlib.redirect_stdout(sys.stderr):
                current = core._sample(
                    pipeline,
                    current,
                    embeds,
                    denoise=settings["denoise"],
                    steps=settings["steps"],
                    seed=seed + pass_index,
                )
    except Exception:
        pipeline = None
        embeds = None
        current = None
        image = None
        _unload_runtime()
        raise

    OUT_DIR.mkdir(exist_ok=True)
    stem = _safe_stem(input_path)
    tag = f"_s{settings['steps']}_d{settings['denoise']:.3f}_p{settings['passes']}"
    outputs: list[dict[str, str]] = []

    if settings["restore"]:
        emit("state", title="恢复细节", message="正在合成处理图的低频与原图的高频细节。")
        with contextlib.redirect_stdout(sys.stderr):
            final_image = core._restore(
                current,
                image,
                sigma=settings["restore_sigma"],
                mode=settings["restore_mode"],
                unsharp_strength=settings["unsharp"],
            )
        mode_tag = "" if settings["restore_mode"] == "gaussian" else f"_{settings['restore_mode']}"
        result_path = _save_png_unique(
            final_image,
            OUT_DIR / f"{stem}_desynth{tag}_r{settings['restore_sigma']:g}{mode_tag}.png"
        )
        outputs.append({"kind": "result", "name": result_path.name, "path": str(result_path)})

        if settings["keep_intermediate"]:
            try:
                intermediate_path = _save_png_unique(current, OUT_DIR / f"{stem}_desynth{tag}.png")
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                emit("log", message=f"最终结果已保存，但中间图保存失败：{exc}")
            else:
                outputs.append(
                    {"kind": "intermediate", "name": intermediate_path.name, "path": str(intermediate_path)}
                )
    else:
        result_path = _save_png_unique(current, OUT_DIR / f"{stem}_desynth{tag}.png")
        outputs.append({"kind": "result", "name": result_path.name, "path": str(result_path)})

    emit("complete", seed=seed, outputs=outputs)


def serve() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("请求格式无效")
            action = request.get("action")
            if action == "run":
                _run(request)
            elif action == "shutdown":
                emit("log", message="正在释放模型并退出")
                _unload_runtime()
                return
            else:
                raise ValueError("未知操作")
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            emit("error", message=str(exc) or exc.__class__.__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="DeSynth Web UI worker")
    parser.add_argument("--check", action="store_true", help="validate local runtime files and exit")
    args = parser.parse_args()
    if args.check:
        checks = environment_checks()
        print(json.dumps({"ready": all(item["ok"] for item in checks), "checks": checks}, ensure_ascii=False))
        return
    serve()


if __name__ == "__main__":
    main()
