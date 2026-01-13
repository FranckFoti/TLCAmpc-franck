from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from string import Template
from typing import Any, NoReturn

import httpx
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image


def load_parametrized_json(
    path: str | Path, params: dict[str, str] | None = None
) -> dict[str, Any]:
    """Load a JSON file and substitute `${var}` placeholders.

    This is intentionally minimal: the file contents are treated as a string.Template.

    Notes:
    - Substitution happens before JSON parsing.
    - If you want to substitute a *number*, pass the literal (e.g. "0.15").
    - If you want to substitute a *string*, include quotes in the template or pass a quoted value.

    Example:
        {"dt": ${dt}}
    with params {"dt": "0.05"}
    """

    text = Path(path).read_text(encoding="utf-8")
    if params:
        text = Template(text).safe_substitute(params)
    return json.loads(text)


def run_live_view(
    *,
    config_path: str | Path,
    params: dict[str, str] | None = None,
    base_url: str = "http://127.0.0.1:8000",
    steps: int = 200,
    step_n: int = 1,
    sleep_s: float = 0.05,
    trace_len: int = 50,
    width: int = 900,
    height: int = 700,
    dpi: int = 120,
    elev: float = 20.0,
    azim: float = -60.0,
    record_dir: str | Path | None = None,
    gif_path: str | Path | None = None,
    gif_fps: float = 20.0,
    timeout_s: float = 10.0,
) -> None:
    """Load scenario config into the REST API and display a live-updating render."""

    cfg = load_parametrized_json(config_path, params=params)

    # Guard against accidentally loading a JSON *string* (which would be sent as a JSON string
    # and trigger FastAPI/Pydantic "Input should be a valid dictionary" 422 errors).
    if isinstance(cfg, str):
        cfg = json.loads(cfg)
    if not isinstance(cfg, dict):
        raise TypeError(f"Config must decode to a JSON object/dict, got {type(cfg).__name__}")

    record_path = Path(record_dir) if record_dir is not None else None
    if record_path is not None:
        record_path.mkdir(parents=True, exist_ok=True)

    gif_out = Path(gif_path) if gif_path is not None else None

    frames: list[Image.Image] = []

    timeout = httpx.Timeout(timeout_s)

    with httpx.Client(timeout=timeout) as client:
        # Be explicit about content-type to avoid servers interpreting the body as text.
        try:
            r = client.post(
                f"{base_url}/config",
                content=json.dumps(cfg),
                headers={"Content-Type": "application/json"},
            )
            r.raise_for_status()
        except httpx.TimeoutException as e:
            raise RuntimeError(
                "Unsolvable configuration (server timed out while loading /config). "
                "Try reducing the number of drones, increasing the MPC horizon, or lowering constraints."
            ) from e

        plt.ion()
        fig, ax = plt.subplots()
        ax.set_axis_off()

        img_artist = None

        for i in range(steps):
            if step_n > 0:
                try:
                    r = client.post(f"{base_url}/step", params={"n": step_n})
                    r.raise_for_status()
                except httpx.TimeoutException as e:
                    raise RuntimeError(
                        f"Unsolvable configuration (server timed out during /step at frame {i})."
                    ) from e

            try:
                r = client.get(
                    f"{base_url}/render",
                    params={
                        "width": width,
                        "height": height,
                        "dpi": dpi,
                        "elev": elev,
                        "azim": azim,
                        "trace_len": trace_len,
                    },
                )
                r.raise_for_status()
            except httpx.TimeoutException as e:
                raise RuntimeError(
                    f"Unsolvable configuration (server timed out during /render at frame {i})."
                ) from e

            # For display
            img = mpimg.imread(io.BytesIO(r.content), format="png")
            if img_artist is None:
                img_artist = ax.imshow(img)
            else:
                img_artist.set_data(img)

            # For recording
            if record_path is not None or gif_out is not None:
                pil_img = Image.open(io.BytesIO(r.content)).convert("RGBA")
                if record_path is not None:
                    frame_path = record_path / f"frame_{i:05d}.png"
                    frame_path.write_bytes(r.content)
                if gif_out is not None:
                    frames.append(pil_img)

            fig.canvas.draw_idle()
            plt.pause(0.001)

            if sleep_s > 0:
                time.sleep(sleep_s)

        plt.ioff()
        plt.show()

    if gif_out is not None:
        if not frames:
            raise RuntimeError("No frames captured; cannot write GIF")
        duration_ms = int(round(1000.0 / max(0.1, float(gif_fps))))
        gif_out.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            gif_out,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )


def _parse_kv_params(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Bad --param '{item}'. Expected KEY=VALUE")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Live-view DroneSim by polling /render while stepping the sim"
    )
    p.add_argument(
        "--config", required=True, help="Path to scenario JSON (supports ${var} placeholders)"
    )
    p.add_argument(
        "--param",
        action="append",
        default=[],
        help="Template parameter KEY=VALUE (may be repeated)",
    )
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--step-n", type=int, default=1)
    p.add_argument("--sleep", type=float, default=0.05)
    p.add_argument("--trace-len", type=int, default=50)

    p.add_argument(
        "--record-dir", default=None, help="If set, write PNG frames into this directory"
    )
    p.add_argument(
        "--gif", dest="gif_path", default=None, help="If set, write an animated GIF to this path"
    )
    p.add_argument("--gif-fps", type=float, default=20.0, help="FPS for the generated GIF")

    p.add_argument("--width", type=int, default=900)
    p.add_argument("--height", type=int, default=700)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--elev", type=float, default=20.0)
    p.add_argument("--azim", type=float, default=-60.0)
    p.add_argument(
        "--timeout",
        dest="timeout_s",
        type=float,
        default=10.0,
        help="HTTP timeout seconds (increase for large centralized MPC problems)",
    )

    args = p.parse_args(argv)

    params = _parse_kv_params(args.param)

    run_live_view(
        config_path=args.config,
        params=params,
        base_url=args.base_url,
        steps=args.steps,
        step_n=args.step_n,
        sleep_s=args.sleep,
        trace_len=args.trace_len,
        width=args.width,
        height=args.height,
        dpi=args.dpi,
        elev=args.elev,
        azim=args.azim,
        record_dir=args.record_dir,
        gif_path=args.gif_path,
        gif_fps=args.gif_fps,
        timeout_s=args.timeout_s,
    )


def _die(msg: str, code: int = 1) -> "NoReturn":
    print(f" ==== {msg}", file=sys.stderr)
    raise SystemExit(code)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _die("Interrupted by user", code=0)
    except RuntimeError as e:
        _die(str(e), code=1)
    except Exception:
        # Unexpected error: include traceback (much more helpful than just str(e))
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
