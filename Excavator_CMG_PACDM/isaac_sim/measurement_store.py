"""Crash-resilient native measurements, independent of Isaac/Omniverse.

A committed chunk contains only measured data. These are evidence checkpoints,
not restart states: PhysX solver/contact caches and hydraulic history are not
restored, and a crashed simulation is never silently resumed or marked complete.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np

STATE_FIELDS = ("time", "positions", "quaternions_wxyz",
                "linear_velocities_origin_world", "angular_velocities_world",
                "kinetic_energy_J", "potential_energy_J", "applied_work_midpoint_J",
                "closure_gaps_m", "gear_errors_rad")
CONTROL_FIELDS = ("time", "efforts", "forces_world", "torques_world_about_com")
CONTACT_FIELDS = ("time", "reported_net_contact_forces_world")
GROUPS = (("state", "states.npz", STATE_FIELDS),
          ("control", "applied_wrenches.npz", CONTROL_FIELDS),
          ("contact", "native_contact_forces.npz", CONTACT_FIELDS))


def _replace(source: Path, destination: Path) -> None:
    # Windows indexers/OneDrive may briefly hold the destination open.
    for attempt in range(6):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (attempt + 1))


def atomic_json(path, data) -> None:
    path = Path(path)
    text = json.dumps(data, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    _replace(tmp, path)


def atomic_npz(path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    _replace(tmp, path)


class CheckpointStore:
    def __init__(self, output, body_names, dt):
        self.output = Path(output)
        self.directory = self.output / "checkpoints"
        self.directory.mkdir(parents=True, exist_ok=True)
        if list(self.directory.glob("chunk_*.npz")):
            raise FileExistsError("Use a fresh output folder; evidence checkpoints cannot resume physics")
        self.names = np.asarray(body_names)
        self.dt = float(dt)
        self.index = 0
        self.last_state_time = None

    def flush(self, rows, controls, contacts, status) -> None:
        """Commit all pending groups together, then release their Python buffers."""
        if not (rows or controls or contacts):
            atomic_json(self.output / "status.json", status)
            return
        payload = {"body_names": self.names, "dt": np.asarray(self.dt)}
        for prefix, entries, fields in (("state", rows, STATE_FIELDS),
                                         ("control", controls, CONTROL_FIELDS),
                                         ("contact", contacts, CONTACT_FIELDS)):
            if entries:
                for field, column in zip(fields, zip(*entries)):
                    payload[prefix + "_" + field] = np.asarray(column, dtype=float) if field == "time" else np.asarray(column)
        snapshot = dict(status)
        snapshot["checkpoint_index"] = self.index
        snapshot["checkpoint_scope"] = "committed native measurements, not a resumable physics state"
        snapshot["checkpoint_state_time_s"] = (float(rows[-1][0]) if rows else self.last_state_time)
        payload["status_json"] = np.asarray(json.dumps(snapshot, allow_nan=False))
        path = self.directory / ("chunk_%06d.npz" % self.index)
        if path.exists():
            raise FileExistsError("Refusing to replace a committed measurement chunk")
        atomic_npz(path, **payload)
        if rows:
            self.last_state_time = float(rows[-1][0])
        self.index += 1
        rows.clear()
        controls.clear()
        contacts.clear()
        status.update({key: snapshot[key] for key in
                       ("checkpoint_index", "checkpoint_scope", "checkpoint_state_time_s")})
        atomic_json(self.output / "progress.json", snapshot)
        atomic_json(self.output / "status.json", status)


def assemble_measurements(output):
    """Build familiar NPZ files in the controller process, with disk-backed arrays.

    Never skips a bad committed chunk or fabricates absent samples. Incomplete
    *.tmp files are ignored. At most one chunk is decompressed at a time while
    filling memory-mapped arrays, avoiding a full-run RAM copy in Isaac.
    """
    output = Path(output)
    chunks = sorted((output / "checkpoints").glob("chunk_*.npz"))
    report = {"chunks": len(chunks), "artifacts": {}, "errors": [],
              "scope": "exact concatenation of committed evidence; not proof of mission success"}
    if not chunks:
        report["note"] = "No committed checkpoints were available"
        atomic_json(output / "measurement_assembly.json", report)
        return report
    if [p.name for p in chunks] != ["chunk_%06d.npz" % i for i in range(len(chunks))]:
        report["errors"].append("Checkpoint numbering has a gap; refusing partial concatenation")
        atomic_json(output / "measurement_assembly.json", report)
        return report
    try:
        with np.load(chunks[0], allow_pickle=False) as data:
            names, dt = data["body_names"], float(data["dt"])
        inventories = {prefix: [] for prefix, _, _ in GROUPS}
        for chunk in chunks:
            with np.load(chunk, allow_pickle=False) as data:
                if not np.array_equal(data["body_names"], names) or float(data["dt"]) != dt:
                    raise ValueError("Checkpoint body order or timestep changed")
                for prefix, _, fields in GROUPS:
                    key = prefix + "_time"
                    if key not in data:
                        continue
                    times = data[key]
                    if (times.ndim != 1 or not len(times) or not np.all(np.isfinite(times))
                            or not np.all(np.diff(times) > 0)):
                        raise ValueError("Invalid checkpoint time array: " + str(chunk))
                    inventory = inventories[prefix]
                    if inventory and float(times[0]) <= inventory[-1][3]:
                        raise ValueError("Overlapping checkpoint times: " + prefix)
                    for field in fields:
                        if prefix + "_" + field not in data:
                            raise ValueError("Missing checkpoint field: " + field)
                    inventory.append((chunk, len(times), float(times[0]), float(times[-1])))
        with np.load(chunks[-1], allow_pickle=False) as data:
            report["last_committed_status"] = json.loads(str(data["status_json"]))
        for prefix, filename, fields in GROUPS:
            inventory = inventories[prefix]
            if not inventory:
                continue
            total = sum(item[1] for item in inventory)
            target = output / filename
            # Temporary NPY files limit RAM, and are removed only after closing
            # all Windows memory mappings. Original checkpoints are retained.
            with tempfile.TemporaryDirectory(prefix=".assemble_", dir=output) as work:
                work = Path(work)
                memmaps = {}
                try:
                    with np.load(inventory[0][0], allow_pickle=False) as data:
                        for field in fields:
                            sample = data[prefix + "_" + field]
                            shape = (total,) + sample.shape[1:]
                            # NumPy cannot memory-map a zero-byte data region.
                            if np.prod(shape) == 0:
                                with (work / (field + ".npy")).open("wb") as stream:
                                    np.save(stream, np.empty(shape, dtype=sample.dtype), allow_pickle=False)
                            else:
                                memmaps[field] = np.lib.format.open_memmap(
                                    work / (field + ".npy"), mode="w+", dtype=sample.dtype, shape=shape)
                    offset = 0
                    for chunk, count, _, _ in inventory:
                        with np.load(chunk, allow_pickle=False) as data:
                            for field in fields:
                                values = data[prefix + "_" + field]
                                if values.shape[0] != count or not np.all(np.isfinite(values)):
                                    raise ValueError("Invalid measurement values: " + field)
                                if field in memmaps:
                                    dest = memmaps[field]
                                    if values.shape[1:] != dest.shape[1:] or values.dtype != dest.dtype:
                                        raise ValueError("Measurement shape/dtype changed: " + field)
                                    dest[offset:offset + count] = values
                        offset += count
                    for mmap in memmaps.values():
                        mmap.flush()
                finally:
                    for mmap in memmaps.values():
                        mmap._mmap.close()
                    memmaps.clear()
                if prefix == "state":
                    np.save(work / "body_names.npy", names, allow_pickle=False)
                if prefix == "contact":
                    np.save(work / "dt.npy", np.asarray(dt), allow_pickle=False)
                tmp = target.with_name(target.name + ".tmp")
                with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
                    for npy in sorted(work.glob("*.npy")):
                        archive.write(npy, npy.name)
                    if prefix == "state":
                        for alias, original in (("linear_velocities", "linear_velocities_origin_world"),
                                                ("angular_velocities", "angular_velocities_world")):
                            archive.write(work / (original + ".npy"), alias + ".npy")
                with tmp.open("r+b") as stream:
                    os.fsync(stream.fileno())
                _replace(tmp, target)
            report["artifacts"][filename] = {"samples": total,
                "first_time_s": inventory[0][2], "last_time_s": inventory[-1][3]}
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
        report["errors"].append("%s: %s" % (type(exc).__name__, exc))
    atomic_json(output / "measurement_assembly.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    result = assemble_measurements(args.result)
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(result["errors"]))
