"""Offline dry run of the real bridge server + native runner loop (no Isaac).

tests/fake_isaac provides a MuJoCo-backed stand-in for the Isaac/PhysX API.
This checks the Python control/measurement path end to end for a few physics
steps, including the q22 gauge report and the grain rolling-resistance torques.
It is not evidence of PhysX behaviour.
"""
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FAKE = ROOT/'tests'/'fake_isaac'
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec('mujoco') is None or importlib.util.find_spec('pinocchio') is None,
    reason='mujoco/pinocchio controller runtime unavailable')


def test_runner_and_bridge_dry_run(tmp_path):
    sys.path.insert(0, str(ROOT))
    from measurement_store import assemble_measurements
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    bridge = subprocess.Popen([sys.executable, '-u', str(ROOT/'bridge_server.py'), '--port', str(port)],
                              cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        assert bridge.stdout.readline().startswith('EXCAVATOR_BRIDGE_READY')
        out = tmp_path/'run'
        env = dict(os.environ, PYTHONPATH=str(FAKE)+os.pathsep+os.environ.get('PYTHONPATH', ''),
                   FAKE_GRAIN_SPIN='2.0')
        runner = subprocess.run([sys.executable, '-u', str(ROOT/'run_isaac.py'),
                                 '--manifest', str(ROOT/'generated/soil_final/manifest.json'),
                                 '--duration', '0.06', '--port', str(port), '--case', 'soil_final',
                                 '--mode', 'mission', '--output', str(out), '--checkpoint-dt', '0.01',
                                 '--require-contact-forces', '--headless'],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
        assert runner.returncode == 0, runner.stdout[-3000:]+runner.stderr[-3000:]
    finally:
        bridge.terminate()
        bridge.wait(timeout=30)
    status = json.loads((out/'status.json').read_text(encoding='utf-8'))
    assert status['status'] == 'NATIVE_RUN_COMPLETED_UNASSESSED', status.get('error')
    assert status['native_steps_completed'] == 60
    rolling = status['soil_rolling_resistance']
    assert rolling['applied'] and rolling['steps_applied'] == 60 and rolling['grain_bodies'] == 105
    gauge = status['bridge_summary']['passive_spin_gauge']['coordinates'][0]
    assert gauge['joint'] == 'q22' and gauge['max_abs_measured_offset_rad'] < 1.5
    assert assemble_measurements(out)['errors'] == []
    manifest = json.loads((ROOT/'generated/soil_final/manifest.json').read_text(encoding='utf-8'))
    grains = [k for k, b in enumerate(manifest['bodies']) if b['name'].startswith('grain_')]
    k = grains[0]  # grain_0000, kicked to 2 rad/s after the first step by the stand-in
    with np.load(out/'applied_wrenches.npz') as wrenches, np.load(out/'states.npz') as states, \
            np.load(out/'native_contact_forces.npz') as contact:
        assert wrenches['time'].shape == (60,)
        torque = wrenches['torques_world_about_com']
        assert np.all(np.isfinite(torque[:, grains]))
        # Grains receive torques only: no forces are ever applied to soil.
        assert not np.any(wrenches['forces_world'][:, grains])
        assert np.any(np.abs(contact['reported_net_contact_forces_world']) > 1e-6)
        # Before landing (3 mm initial gap) the grain has no contact and gets no torque.
        assert not np.any(torque[5, k])
        # Exact law at control step 50: contact force and spin recorded at t = 0.050 s.
        i_state = int(np.flatnonzero(np.isclose(states['time'], .050))[0])
        i_contact = int(np.flatnonzero(np.isclose(contact['time'], .050))[0])
        w = states['angular_velocities_world'][i_state, k]
        force = contact['reported_net_contact_forces_world'][i_contact, k].astype(float)
        f = np.linalg.norm(force)
        n = force/f
        w_n = np.dot(w, n)*n
        w_t = w-w_n
        cap = .5*max(manifest['bodies'][k]['inertia_diagonal'])/.001
        expected = (-min(.008*f/max(np.linalg.norm(w_t), .5), cap)*w_t
                    - min(.003*f/max(np.linalg.norm(w_n), .5), cap)*w_n)
        assert np.linalg.norm(w) > .01 and f > 1.
        assert np.allclose(torque[50, k], expected, rtol=1e-5, atol=1e-9)
    assert rolling['peak_torque_Nm'] > 0. and rolling['applied_torque_work_J'] < 0.


def _dry_run(tmp_path, extra_env, duration='0.02', extra_args=()):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    bridge = subprocess.Popen([sys.executable, '-u', str(ROOT/'bridge_server.py'), '--port', str(port)],
                              cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        assert bridge.stdout.readline().startswith('EXCAVATOR_BRIDGE_READY')
        out = tmp_path/'run'
        env = dict(os.environ, PYTHONPATH=str(FAKE)+os.pathsep+os.environ.get('PYTHONPATH', ''), **extra_env)
        runner = subprocess.run([sys.executable, '-u', str(ROOT/'run_isaac.py'),
                                 '--manifest', str(ROOT/'generated/soil_final/manifest.json'),
                                 '--duration', duration, '--port', str(port), '--output', str(out),
                                 '--checkpoint-dt', '0.01', '--headless', *extra_args],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
        assert runner.returncode == 0, runner.stdout[-3000:]+runner.stderr[-3000:]
    finally:
        bridge.terminate()
        bridge.wait(timeout=30)
    return json.loads((out/'status.json').read_text(encoding='utf-8'))


def test_lost_contact_view_disables_rolling_instead_of_aborting(tmp_path):
    """Without --require-contact-forces a lost readback must not stop the run."""
    status = _dry_run(tmp_path, {'FAKE_CONTACT_VIEW_FAIL_AFTER_FIRST': '1'})
    assert status['status'] == 'NATIVE_RUN_COMPLETED_UNASSESSED', status.get('error')
    assert status['native_steps_completed'] == 20
    rolling = status['soil_rolling_resistance']
    assert rolling['applied'] is False and 'disabled after native step 1' in rolling['note']
