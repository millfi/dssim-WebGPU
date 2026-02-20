#!/usr/bin/env python3
import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple


def _get_nested(obj, *keys):
    cur = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _score_text(obj):
    value = _get_nested(obj, "result", "score_text")
    if value is None:
        value = obj.get("score_text")
    return value


def _score_bits(obj):
    value = _get_nested(obj, "result", "score_bits_u64")
    if value is None:
        value = obj.get("score_bits_u64")
    if value is None:
        return None
    if isinstance(value, int):
        return value & ((1 << 64) - 1)
    text = str(value).strip().lower()
    if text.startswith("0x"):
        return int(text, 16)
    return int(text)


def _score_float(obj):
    value = _get_nested(obj, "result", "score_f64")
    if value is None:
        value = obj.get("score_f64")
    if value is None:
        return None
    return float(value)


def _bits_from_float64(value):
    packed = struct.pack(">d", float(value))
    return struct.unpack(">Q", packed)[0]


def _bits_from_float32(value):
    packed = struct.pack(">f", float(value))
    return struct.unpack(">I", packed)[0]


def _format_bits(bits):
    return f"0x{bits:016X}"


def _load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _status(obj):
    return obj.get("status")


def _input_pair(obj):
    return (
        _get_nested(obj, "input", "image1"),
        _get_nested(obj, "input", "image2"),
    )


def compare_scores(ref_obj, gpu_obj):
    issues = []

    ref_status = _status(ref_obj)
    gpu_status = _status(gpu_obj)
    if ref_status and ref_status != "ok":
        issues.append(f"Reference JSON status is not ok: {ref_status}")
    if gpu_status and gpu_status != "ok":
        issues.append(f"GPU JSON status is not ok: {gpu_status}")

    ref_pair = _input_pair(ref_obj)
    gpu_pair = _input_pair(gpu_obj)
    if all(v is not None for v in ref_pair) and all(v is not None for v in gpu_pair):
        if ref_pair != gpu_pair:
            issues.append(
                "Input pair mismatch: "
                f"ref=({ref_pair[0]}, {ref_pair[1]}), gpu=({gpu_pair[0]}, {gpu_pair[1]})"
            )

    ref_text = _score_text(ref_obj)
    gpu_text = _score_text(gpu_obj)

    if ref_text is not None and gpu_text is not None:
        if str(ref_text) != str(gpu_text):
            issues.append(
                "score_text mismatch (EXACT required): "
                f"ref={ref_text}, gpu={gpu_text}"
            )
        return issues

    ref_bits = _score_bits(ref_obj)
    gpu_bits = _score_bits(gpu_obj)
    if ref_bits is not None and gpu_bits is not None:
        if ref_bits != gpu_bits:
            issues.append(
                "score_bits_u64 mismatch: "
                f"ref={_format_bits(ref_bits)}, gpu={_format_bits(gpu_bits)}"
            )
        return issues

    ref_float = _score_float(ref_obj)
    gpu_float = _score_float(gpu_obj)
    if ref_float is not None and gpu_float is not None:
        ref_float_bits = _bits_from_float64(ref_float)
        gpu_float_bits = _bits_from_float64(gpu_float)
        if ref_float_bits != gpu_float_bits:
            issues.append(
                "score_f64 bit mismatch: "
                f"ref={_format_bits(ref_float_bits)}, gpu={_format_bits(gpu_float_bits)}"
            )
        return issues

    issues.append(
        "No comparable score fields found. Provide result.score_text or result.score_bits_u64."
    )
    return issues


def _resolve_dump_entry(obj: dict, key: str) -> Optional[dict]:
    entry = _get_nested(obj, "debug_dumps", key)
    if entry is None:
        return None
    if isinstance(entry, str):
        return {"path": entry}
    if isinstance(entry, dict):
        return entry
    return None


def _dtype_size(dtype: str) -> int:
    if dtype == "u8":
        return 1
    if dtype == "u32_le":
        return 4
    if dtype == "f32_le":
        return 4
    if dtype == "f64_le":
        return 8
    raise ValueError(f"unsupported dtype: {dtype}")


def _read_buffer(path: Path, dtype: str) -> List[Any]:
    raw = path.read_bytes()
    item_size = _dtype_size(dtype)
    if len(raw) % item_size != 0:
        raise ValueError(
            f"buffer size is not aligned with dtype ({dtype}): bytes={len(raw)}, item_size={item_size}"
        )

    if dtype == "u8":
        return list(raw)
    if dtype == "u32_le":
        count = len(raw) // 4
        return list(struct.unpack("<" + "I" * count, raw))
    if dtype == "f32_le":
        count = len(raw) // 4
        return list(struct.unpack("<" + "f" * count, raw))
    if dtype == "f64_le":
        count = len(raw) // 8
        return list(struct.unpack("<" + "d" * count, raw))
    raise ValueError(f"unsupported dtype: {dtype}")


def _first_mismatch(ref_values: List[Any], gpu_values: List[Any], dtype: str) -> Optional[Tuple[int, Any, Any]]:
    limit = min(len(ref_values), len(gpu_values))
    for i in range(limit):
        rv = ref_values[i]
        gv = gpu_values[i]
        if dtype == "f32_le":
            if _bits_from_float32(rv) != _bits_from_float32(gv):
                return (i, rv, gv)
        elif dtype == "f64_le":
            if _bits_from_float64(rv) != _bits_from_float64(gv):
                return (i, rv, gv)
        else:
            if rv != gv:
                return (i, rv, gv)
    return None


def compare_debug_buffer(
    ref_obj: dict,
    gpu_obj: dict,
    buffer_key: str,
    force_dtype: Optional[str] = None,
) -> List[str]:
    issues: List[str] = []
    ref_entry = _resolve_dump_entry(ref_obj, buffer_key)
    gpu_entry = _resolve_dump_entry(gpu_obj, buffer_key)

    if ref_entry is None:
        issues.append(f"ref JSON missing debug_dumps.{buffer_key}")
        return issues
    if gpu_entry is None:
        issues.append(f"gpu JSON missing debug_dumps.{buffer_key}")
        return issues

    ref_path_str = ref_entry.get("path")
    gpu_path_str = gpu_entry.get("path")
    if not ref_path_str or not gpu_path_str:
        issues.append(f"debug_dumps.{buffer_key}.path missing in ref or gpu JSON")
        return issues

    dtype = force_dtype or ref_entry.get("elem_type") or gpu_entry.get("elem_type") or "u8"
    if ref_entry.get("elem_type") and gpu_entry.get("elem_type"):
        if ref_entry.get("elem_type") != gpu_entry.get("elem_type") and force_dtype is None:
            issues.append(
                "elem_type mismatch between ref and gpu dumps: "
                f"ref={ref_entry.get('elem_type')} gpu={gpu_entry.get('elem_type')}"
            )
            return issues

    ref_path = Path(ref_path_str)
    gpu_path = Path(gpu_path_str)
    if not ref_path.exists():
        issues.append(f"ref dump file not found: {ref_path}")
        return issues
    if not gpu_path.exists():
        issues.append(f"gpu dump file not found: {gpu_path}")
        return issues

    try:
        ref_values = _read_buffer(ref_path, dtype)
        gpu_values = _read_buffer(gpu_path, dtype)
    except Exception as exc:
        issues.append(f"failed to read debug buffer ({dtype}): {exc}")
        return issues

    if len(ref_values) != len(gpu_values):
        issues.append(
            "buffer length mismatch: "
            f"ref={len(ref_values)} gpu={len(gpu_values)} elements"
        )
        mismatch = _first_mismatch(ref_values, gpu_values, dtype)
        if mismatch is not None:
            idx, rv, gv = mismatch
            issues.append(f"first mismatch index={idx}, ref={rv}, gpu={gv}")
        return issues

    mismatch = _first_mismatch(ref_values, gpu_values, dtype)
    if mismatch is not None:
        idx, rv, gv = mismatch
        if dtype == "f32_le":
            issues.append(
                "debug buffer mismatch: "
                f"index={idx}, ref={rv} (0x{_bits_from_float32(rv):08X}), "
                f"gpu={gv} (0x{_bits_from_float32(gv):08X})"
            )
        elif dtype == "f64_le":
            issues.append(
                "debug buffer mismatch: "
                f"index={idx}, ref={rv} (0x{_bits_from_float64(rv):016X}), "
                f"gpu={gv} (0x{_bits_from_float64(gv):016X})"
            )
        else:
            issues.append(f"debug buffer mismatch: index={idx}, ref={rv}, gpu={gv}")
        return issues

    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Compare reference and GPU DSSIM JSON outputs with exact-match policy."
    )
    parser.add_argument("ref_json", type=Path)
    parser.add_argument("gpu_json", type=Path)
    parser.add_argument(
        "--buffer-key",
        help="Compare intermediate buffer in debug_dumps.<key> (e.g., stage0_absdiff_u32le).",
    )
    parser.add_argument(
        "--buffer-dtype",
        choices=["u8", "u32_le", "f32_le", "f64_le"],
        help="Force dtype for --buffer-key comparison.",
    )
    parser.add_argument(
        "--buffer-only",
        action="store_true",
        help="Skip final score comparison and compare only --buffer-key.",
    )
    args = parser.parse_args()

    try:
        ref_obj = _load_json(args.ref_json)
        gpu_obj = _load_json(args.gpu_json)
    except Exception as exc:
        print(f"[compare] failed to read JSON: {exc}", file=sys.stderr)
        return 2

    issues = []
    if not args.buffer_only:
        issues.extend(compare_scores(ref_obj, gpu_obj))
    if args.buffer_key:
        issues.extend(
            compare_debug_buffer(
                ref_obj=ref_obj,
                gpu_obj=gpu_obj,
                buffer_key=args.buffer_key,
                force_dtype=args.buffer_dtype,
            )
        )
    elif args.buffer_only:
        issues.append("--buffer-only requires --buffer-key")

    if issues:
        print("[compare] FAIL")
        for issue in issues:
            print(f" - {issue}")
        return 1

    if args.buffer_only and args.buffer_key:
        print(f"[compare] PASS (exact match, buffer-only '{args.buffer_key}')")
    elif args.buffer_key:
        print(f"[compare] PASS (exact match, score + buffer '{args.buffer_key}')")
    else:
        print("[compare] PASS (exact match)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
