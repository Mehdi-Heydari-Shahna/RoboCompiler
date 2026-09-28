"""Strict, schema-stable exchange reader for the supplied excavator CMG.

This module does not import MATLAB, PACDM or a physics backend.  It restores only
declared collection/matrix shapes in the v0.3 MATLAB JSON export, retaining all
numeric values and findings.  Validation establishes serialization and basic
physical-record consistency, not measured geometry or valid dynamics.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile

import numpy as np
from jsonschema import Draft202012Validator


EXCHANGE_SCHEMA = "excavator.cmg.exchange/0.4"
SCHEMA_PATH = Path(__file__).with_name("cmg.schema.json")


class CMGValidationError(ValueError):
    """Unsupported, ambiguous or inconsistent CMG exchange data."""


def _require(condition, message):
    if not condition:
        raise CMGValidationError(message)


def _reject_constant(value):
    raise CMGValidationError(f"Nonfinite JSON token is not allowed: {value}")


def _object_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def _read_json(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream, parse_constant=_reject_constant,
                         object_pairs_hook=_object_without_duplicate_keys)


def _finite_tree(value, path=""):
    if isinstance(value, float):
        _require(math.isfinite(value), f"{path}: nonfinite number")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{path}/{index}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{path}/{key}")


def migrate_cmg(data):
    """Return (canonical_copy, change_list); never mutate supplied records.

    Supported migration is only the known v0.3 interpretation of the v0.2
    excavator physical snapshot.  Already canonical input is validated and
    copied with an empty new change list.  Malformed canonical input is never
    silently repaired.  Changes retain explicit paths and original shapes.
    """
    _require(isinstance(data, dict), "CMG root must be an object")
    if "exchange" in data:
        validate_cmg(data)
        return deepcopy(data), []
    _require(data.get("schema") == "excavator.roboir.prototype/0.2",
             "Unsupported physical source schema")
    _require(isinstance(data.get("interpretation"), dict)
             and data["interpretation"].get("schema") == "excavator.cmg.interpretation/0.3",
             "Migration requires the v0.3 CMG interpretation")
    _finite_tree(data)
    result = deepcopy(data)
    changes = []

    def collection(record, key, path):
        _require(key in record, f"{path}/{key}: missing declared collection")
        value = record[key]
        if not isinstance(value, list):
            _require(value is not None and not isinstance(value, bool),
                     f"{path}/{key}: unsupported collection value")
            record[key] = [value]
            changes.append({"path": f"{path}/{key}",
                            "operation": "wrap_singleton_collection",
                            "original_kind": type(value).__name__,
                            "canonical_shape": [1]})

    def matrix(record, key, rows, columns, path):
        _require(key in record, f"{path}/{key}: missing matrix")
        value = record[key]
        array = np.asarray(value)
        shape = (rows, columns)
        if array.shape == shape:
            return
        # MATLAB collapses single-row/single-column matrices to vectors.  Never
        # reshape an arbitrary multidimensional array by element count alone.
        allowed = ((columns == 1 and array.shape == (rows,))
                   or (rows == 1 and array.shape == (columns,))
                   or (rows == columns == 1 and array.shape == ()))
        _require(allowed, f"{path}/{key}: cannot infer matrix shape {shape} from {array.shape}")
        record[key] = array.reshape(shape).tolist()
        changes.append({"path": f"{path}/{key}",
                        "operation": "restore_matrix_shape",
                        "original_kind": "scalar" if array.ndim == 0 else "vector",
                        "canonical_shape": list(shape)})

    def partition(record, path):
        for key in ("coordinate_ids", "reference_q_SI", "closure_cycle_ids",
                    "preferred_ids", "derived_ids", "independent_ids", "passive_ids",
                    "selected_constraint_rows"):
            collection(record, key, path)
        if "constraint_singular_values" in record:
            collection(record, "constraint_singular_values", path)
        columns = len(record["independent_ids"])
        matrix(record, "differential_map", len(record["passive_ids"]), columns, path)
        matrix(record, "tangent_map", len(record["coordinate_ids"]), columns, path)

    for key in ("bodies", "joints", "primary_coordinate_ids", "modules", "reference_poses",
                "actuators", "source_findings", "pending"):
        collection(result, key, "")
    collection(result["provenance"], "source_files", "/provenance")
    for index, joint in enumerate(result["joints"]):
        for key in ("source_base_rt_records", "source_follower_rt_records"):
            collection(joint, key, f"/joints/{index}")
    for key in ("joint_ids", "coefficients"):
        collection(result["internal_mode"], key, "/internal_mode")
    for index, module in enumerate(result["modules"]):
        for key in ("joint_ids", "independent_ids"):
            collection(module, key, f"/modules/{index}")
    for index, reference in enumerate(result["reference_poses"]):
        for key in ("coordinate_ids", "q_SI"):
            collection(reference, key, f"/reference_poses/{index}")
    collection(result["boom_reference"], "coordinate_ids", "/boom_reference")
    command = result["command_map"]
    for key in ("coordinate_ids", "channel_ids", "channels", "unconnected_effort_joint_ids"):
        collection(command, key, "/command_map")
    matrix(command, "B_SI", len(command["coordinate_ids"]), len(command["channel_ids"]), "/command_map")
    for index, channel in enumerate(command["channels"]):
        collection(channel, "branches", f"/command_map/channels/{index}")
    interpretation = result["interpretation"]
    for key in ("modules", "bridge_joint_ids", "articulation_body_ids", "unresolved_findings"):
        collection(interpretation, key, "/interpretation")
    for key in ("coordinate_ids", "q_SI"):
        collection(interpretation["reference"], key, "/interpretation/reference")
    partition(interpretation["full_partition"], "/interpretation/full_partition")
    for index, module in enumerate(interpretation["modules"]):
        path = f"/interpretation/modules/{index}"
        for key in ("joint_ids", "body_ids"):
            collection(module, key, path)
        partition(module, path)
    result["exchange"] = {
        "schema": EXCHANGE_SCHEMA,
        "profile": "komatsu-source-snapshot",
        "migration": {"source_physical_schema": result["schema"],
                      "source_interpretation_schema": interpretation["schema"],
                      "method": "known-shape normalization only",
                      "changes": deepcopy(changes)}}
    validate_cmg(result)
    return result, changes


def _ids(records, label):
    values = [record["id"] for record in records]
    _require(len(values) == len(set(values)), f"{label}: duplicate id")
    return set(values)


def _matrix_shape(value, rows, columns, label):
    _require(np.asarray(value).shape == (rows, columns),
             f"{label}: expected matrix shape {(rows, columns)}")


def _transform(value, label):
    transform = np.asarray(value, dtype=float)
    rotation = transform[:3, :3]
    _require(np.allclose(transform[3], [0, 0, 0, 1], rtol=0, atol=1e-12),
             f"{label}: invalid homogeneous last row")
    _require(np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-9)
             and abs(np.linalg.det(rotation) - 1) < 1e-9,
             f"{label}: rotation must be proper orthonormal")


def _partition(record, joint_ids, label):
    coordinates = record["coordinate_ids"]
    active, passive = record["independent_ids"], record["passive_ids"]
    _require(set(coordinates) <= joint_ids, f"{label}: unknown moving coordinate")
    _require(set(active).isdisjoint(passive) and set(active + passive) == set(coordinates),
             f"{label}: independent/passive partition is incomplete or overlapping")
    _require(set(record["preferred_ids"]).isdisjoint(record["derived_ids"])
             and record["preferred_ids"] + record["derived_ids"] == active,
             f"{label}: preferred and derived coordinates do not form independent list")
    _require(len(coordinates) == len(record["reference_q_SI"]), f"{label}: coordinate/value length mismatch")
    _require(record["local_dof"] == len(active)
             and record["constraint_rank"] == len(passive), f"{label}: dimension/rank declaration mismatch")
    rows = record["selected_constraint_rows"]
    _require(len(rows) == len(passive)
             and all(row <= 6 * len(record["closure_cycle_ids"]) for row in rows),
             f"{label}: selected row count or bounds invalid")
    _matrix_shape(record["differential_map"], len(passive), len(active), label + "/differential_map")
    _matrix_shape(record["tangent_map"], len(coordinates), len(active), label + "/tangent_map")
    tangent = np.asarray(record["tangent_map"])
    _require(np.allclose(tangent[[coordinates.index(q) for q in active]], np.eye(len(active)), rtol=0, atol=1e-12),
             f"{label}: independent tangent rows are not identity")
    _require(np.allclose(tangent[[coordinates.index(q) for q in passive]], record["differential_map"], rtol=0, atol=1e-12),
             f"{label}: tangent/passive-map disagreement")


def validate_cmg(data):
    """Validate the canonical exchange, raising CMGValidationError on failure.

    No migration occurs here.  Physical rank, closure and backend behavior must
    be evaluated by separate numerical gates; metadata is not proof of them.
    """
    _finite_tree(data)
    schema = _read_json(SCHEMA_PATH)
    errors = sorted(Draft202012Validator(schema).iter_errors(data), key=lambda e: str(list(e.path)))
    if errors:
        error = errors[0]
        path = "/" + "/".join(str(item) for item in error.path)
        raise CMGValidationError(f"Schema {path}: {error.message}")
    bodies, joints = data["bodies"], data["joints"]
    body_ids, joint_ids = _ids(bodies, "bodies"), _ids(joints, "joints")
    moving = {j["id"] for j in joints if j["type"] != "fixed"}
    joint_by_id = {j["id"]: j for j in joints}
    _require(data["root_body"] in body_ids, "Unknown root body")
    for body in bodies:
        inertia = np.asarray(body["inertia_com_kg_m2"])
        _require(np.allclose(inertia, inertia.T, rtol=0, atol=1e-10), f"{body['id']}: inertia is not symmetric")
        eigenvalues = np.linalg.eigvalsh(inertia)
        scale = max(1.0, np.linalg.norm(inertia, 2))
        if body["kind"] == "rigid_body":
            _require(body["mass_kg"] > 0 and eigenvalues[0] > 0,
                     f"{body['id']}: positive mass and definite inertia required for supported source")
            _require(eigenvalues[2] <= eigenvalues[:2].sum() + 1e-10 * scale,
                     f"{body['id']}: inertia violates principal-moment triangle inequality")
            _require(body["source_solid_record"] >= 1 and bool(body["source_solid_id"]),
                     f"{body['id']}: missing CAD solid provenance")
        else:
            _require(body["id"] == data["root_body"] and body["mass_kg"] == 0
                     and np.count_nonzero(inertia) == 0, "Only the massless world reference is supported")
    neighbors = {body: set() for body in body_ids}
    for joint in joints:
        name, base, follower = joint["id"], joint["base_body"], joint["follower_body"]
        _require(base in body_ids and follower in body_ids and base != follower,
                 f"{name}: unknown or identical endpoint bodies")
        neighbors[base].add(follower)
        neighbors[follower].add(base)
        _transform(joint["T_BJ"], name + "/T_BJ")
        _transform(joint["T_FJ"], name + "/T_FJ")
        unit = {"fixed": "none", "prismatic": "m", "revolute": "rad"}[joint["type"]]
        _require(joint["coordinate_unit"] == unit, f"{name}: wrong coordinate unit")
        expected_axis = [0, 0, 0] if joint["type"] == "fixed" else [0, 0, 1]
        _require(np.array_equal(joint["axis"], expected_axis), f"{name}: unsupported source joint axis")
        _require(bool(joint["source_base_rt_records"]), f"{name}: missing source transform reference")
        if joint["type"] != "fixed":
            _require(bool(joint["source_follower_rt_records"]) and bool(joint["source_sid"]),
                     f"{name}: missing follower/source provenance")
    visited, stack = set(), [data["root_body"]]
    while stack:
        body = stack.pop()
        if body not in visited:
            visited.add(body)
            stack.extend(neighbors[body] - visited)
    _require(visited == body_ids, "Physical graph is disconnected")
    expected_counts = {
        "physical_body_instances": sum(b["kind"] == "rigid_body" for b in bodies),
        "unique_solid_records": len({b["source_solid_record"] for b in bodies if b["kind"] == "rigid_body"}),
        "active_revolute": sum(j["type"] == "revolute" for j in joints),
        "active_prismatic": sum(j["type"] == "prismatic" for j in joints),
        "fixed_edges": len(joints) - len(moving), "vertices_including_world": len(bodies),
        "edges_including_fixed": len(joints), "connected_components": 1,
        "cycle_rank": len(joints) - len(bodies) + 1}
    _require(data["source_counts"] == expected_counts, "Source graph counts disagree with records")
    source_names = [s["filename"] for s in data["provenance"]["source_files"]]
    _require(len(source_names) == len(set(source_names)), "Duplicate source filename")
    _require(set(data["primary_coordinate_ids"]) <= moving, "Unknown primary coordinate")
    for reference in data["reference_poses"] + [data["interpretation"]["reference"]]:
        _require(set(reference["coordinate_ids"]) == moving and len(reference["q_SI"]) == len(moving),
                 f"{reference['id']}: full reference coordinates incomplete")
    _ids(data["reference_poses"], "reference_poses")
    for module in data["modules"]:
        _require(set(module["joint_ids"]) <= joint_ids
                 and set(module["independent_ids"]) <= set(module["joint_ids"])
                 and module["root_body"] in body_ids, "Invalid legacy module reference")
    _ids(data["modules"], "legacy modules")
    boom = data["boom_reference"]
    _require(set(boom["coordinate_ids"]) <= moving, "Unknown boom coordinate")
    _matrix_shape(boom["q_SI"], len(boom["q_SI"]), len(boom["coordinate_ids"]), "boom_reference/q_SI")
    command = data["command_map"]
    coordinates, channels = command["coordinate_ids"], command["channel_ids"]
    _require(set(coordinates) == moving, "Command map coordinates incomplete")
    _matrix_shape(command["B_SI"], len(coordinates), len(channels), "command_map/B_SI")
    _require(_ids(command["channels"], "channels") == set(channels), "Command channel records incomplete")
    assembled = np.zeros((len(coordinates), len(channels)))
    connected_branches = {}
    for channel in command["channels"]:
        index = channels.index(channel["id"])
        _require(channel["index"] == index + 1, "Channel index/order mismatch")
        for branch in channel["branches"]:
            q = branch["joint_id"]
            _require(q in moving and q not in connected_branches, "Unknown or duplicate command branch joint")
            expected_unit = "N" if joint_by_id[q]["type"] == "prismatic" else "N*m"
            _require(branch["unit"] == expected_unit, "Effort branch unit mismatch")
            assembled[coordinates.index(q), index] += branch["gain"]
            connected_branches[q] = (index + 1, branch["gain"], branch["converter_sid"])
    _require(np.array_equal(assembled, command["B_SI"]), "Command branches and B_SI disagree")
    _ids(data["actuators"], "actuators")
    drives = [a["drive_joint_id"] for a in data["actuators"]]
    _require(len(drives) == len(set(drives)), "Duplicate driven joint record")
    unconnected = []
    for actuator in data["actuators"]:
        q = actuator["drive_joint_id"]
        _require(q in moving, "Unknown actuator coordinate")
        linear = joint_by_id[q]["type"] == "prismatic"
        _require(actuator["type"] == ("linear_force" if linear else "rotary_torque")
                 and actuator["effort_unit"] == ("N" if linear else "N*m"), "Actuator type/unit mismatch")
        if actuator["input_status"] == "unconnected":
            unconnected.append(q)
            _require(actuator["unknown_runtime_effort"] and actuator["source_command_channel"] == 0
                     and actuator["source_signal_gain"] == 0 and q not in connected_branches,
                     "Unconnected effort must remain unknown, with no known-channel contribution")
        else:
            actual = (actuator["source_command_channel"], actuator["source_signal_gain"], actuator["source_converter_sid"])
            _require(not actuator["unknown_runtime_effort"] and connected_branches.get(q) == actual,
                     "Actuator/command branch mismatch")
    _require(set(drives) == set(connected_branches) | set(unconnected), "Actuator records incomplete")
    _require(set(command["unconnected_effort_joint_ids"]) == set(unconnected) == {"p0"},
             "Supported source must retain the unresolved p0 effort finding")
    interpretation = data["interpretation"]
    _partition(interpretation["full_partition"], moving, "full_partition")
    full = interpretation["full_partition"]
    _require(set(full["coordinate_ids"]) == moving and full["local_dof"] == 7,
             "Supported source requires the seven-coordinate full partition")
    _require(full["preferred_ids"] == data["primary_coordinate_ids"] and len(full["derived_ids"]) == 1,
             "Full preferred/internal coordinate structure mismatch")
    pin = data["internal_mode"]
    _require(pin["body_id"] in body_ids and set(pin["joint_ids"]) == {"q15", "q18", "q22"}
             and len(pin["coefficients"]) == len(pin["joint_ids"])
             and dict(zip(pin["joint_ids"], pin["coefficients"])) == {"q15": 1, "q18": 1, "q22": -1}
             and full["derived_ids"][0] in pin["joint_ids"]
             and pin["local_configuration_dof"] == 7 and pin["passive_internal_modes"] == 1,
             "Supported source internal pin freedom must remain explicit")
    _ids(interpretation["modules"], "interpreted modules")
    covered = set()
    for module in interpretation["modules"]:
        name = module["id"]
        _partition(module, moving, name)
        ids = set(module["joint_ids"])
        _require(ids <= joint_ids and not ids & covered, f"{name}: invalid or overlapping module edges")
        covered |= ids
        endpoints = {joint_by_id[q][k] for q in ids for k in ("base_body", "follower_body")}
        _require(set(module["body_ids"]) == endpoints, f"{name}: module body references disagree")
        expected = {"body": len(endpoints), "joint": len(ids), "moving_joint": len(ids & moving),
                    "cycles": len(ids) - len(endpoints) + 1}
        _require(module["counts"] == expected and len(module["closure_cycle_ids"]) == expected["cycles"],
                 f"{name}: module count declarations disagree")
        _require(set(module["coordinate_ids"]) == ids & moving, f"{name}: module coordinates incomplete")
    bridges = set(interpretation["bridge_joint_ids"])
    _require(not covered & bridges and covered | bridges == joint_ids, "Modules and bridges do not partition physical edges")
    _require(set(interpretation["articulation_body_ids"]) <= body_ids, "Unknown articulation body")
    expected_interpretation_counts = {"body": len(bodies), "joint": len(joints),
                                      "moving_joint": len(moving), "cycles": expected_counts["cycle_rank"]}
    _require(interpretation["physical_graph_counts"] == expected_interpretation_counts,
             "Interpretation/source counts disagree")
    _require(interpretation["local_dof_from_blocks"] == sum(m["local_dof"] for m in interpretation["modules"]) + len(bridges & moving) == 7,
             "Module/bridge mobility declarations disagree")
    _require(_ids(data["source_findings"], "source findings") == {"extra_pin_spin", "unconnected_tilt_effort"},
             "Source findings missing")
    _require(data["source_findings"] == interpretation["unresolved_findings"], "Unresolved source findings changed")
    return None


def load_cmg(path, migrate_legacy=False):
    """Load a strict v0.4 canonical CMG; legacy migration requires opt-in."""
    data = _read_json(path)
    if migrate_legacy and "exchange" not in data:
        data, _ = migrate_cmg(data)
    validate_cmg(data)
    return data


def write_cmg(path, data):
    """Validate and atomically write canonical UTF-8 JSON without NaN/Inf."""
    validate_cmg(data)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent,
                                         prefix=destination.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
