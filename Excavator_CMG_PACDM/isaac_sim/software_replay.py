#!/usr/bin/env python3
"""CPU software renderer for measured native body poses (no Isaac, no GPU).

This renders the SAME saved PhysX body poses as the Isaac replay worker, but in
the controller environment with NumPy only. Geometry, colours and body-relative
geom poses come from the original MuJoCo source model that the USD scene was
exported from; every body is placed at its measured native world pose. No
physics, interpolation, reference trajectory or MuJoCo kinematics is used for
placement. The image is a flat-shaded z-buffer rendering, not RTX.

The Isaac RTX replay workers (``--video-renderer isaac``) can crash inside Kit
(usdrt.population access violations, heap corruption or a Replicator fail-fast).
This renderer does not use Kit, so a movie can be produced from any saved run.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

RENDERER_ID = 'cpu_zbuffer_v1'
SKY_TOP = np.array([.72, .80, .88])
SKY_BOTTOM = np.array([.93, .94, .95])
LIGHT = np.array([.35, -.45, .82])
LIGHT = LIGHT/np.linalg.norm(LIGHT)
FONT_5X7 = {
    '0': ('01110', '10001', '10011', '10101', '11001', '10001', '01110'),
    '1': ('00100', '01100', '00100', '00100', '00100', '00100', '01110'),
    '2': ('01110', '10001', '00001', '00010', '00100', '01000', '11111'),
    '3': ('11110', '00001', '00001', '01110', '00001', '00001', '11110'),
    '4': ('00010', '00110', '01010', '10010', '11111', '00010', '00010'),
    '5': ('11111', '10000', '11110', '00001', '00001', '10001', '01110'),
    '6': ('00110', '01000', '10000', '11110', '10001', '10001', '01110'),
    '7': ('11111', '00001', '00010', '00100', '01000', '01000', '01000'),
    '8': ('01110', '10001', '10001', '01110', '10001', '10001', '01110'),
    '9': ('01110', '10001', '10001', '01111', '00001', '00010', '01100'),
    '.': ('00000', '00000', '00000', '00000', '00000', '01100', '01100'),
    '=': ('00000', '00000', '11111', '00000', '11111', '00000', '00000'),
    ' ': ('00000',)*7,
    't': ('01000', '01000', '11100', '01000', '01000', '01001', '00110'),
    's': ('00000', '00000', '01111', '10000', '01110', '00001', '11110'),
    'e': ('00000', '00000', '01110', '10001', '11111', '10000', '01110'),
    'p': ('00000', '00000', '11110', '10001', '11110', '10000', '10000'),
}


def _quat_matrix(q):
    q = np.asarray(q, dtype=float)
    q = q/np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], axis=-1).reshape(q.shape[:-1]+(3, 3))


def _icosphere(level=1):
    t = (1+5**.5)/2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                  [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
                  [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
                  [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    tris = v[f]
    tris /= np.linalg.norm(tris, axis=2, keepdims=True)
    for _ in range(level):
        a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
        ab, bc, ca = (a+b)/2, (b+c)/2, (c+a)/2
        tris = np.concatenate([np.stack(x, 1) for x in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
        tris /= np.linalg.norm(tris, axis=2, keepdims=True)
    return tris


def _grid_quads(p0, du, dv, nu, nv):
    """Triangles of an nu x nv subdivision of the parallelogram p0 + s du + t dv."""
    s = np.linspace(0., 1., nu+1); t = np.linspace(0., 1., nv+1)
    tris, parity = [], []
    for i in range(nu):
        for j in range(nv):
            a = p0+s[i]*du+t[j]*dv; b = p0+s[i+1]*du+t[j]*dv
            c = p0+s[i+1]*du+t[j+1]*dv; d = p0+s[i]*du+t[j+1]*dv
            tris += [(a, b, c), (a, c, d)]
            parity += [(i+j) % 2]*2
    return np.asarray(tris, float), np.asarray(parity)


def _box(half, tile):
    half = np.asarray(half, float)
    tris, parity = [], []
    for axis in range(3):
        u_axis, v_axis = [k for k in range(3) if k != axis]
        for sign in (-1., 1.):
            p0 = np.zeros(3); p0[axis] = sign*half[axis]
            p0[u_axis] = -half[u_axis]; p0[v_axis] = -half[v_axis]
            du = np.zeros(3); du[u_axis] = 2*half[u_axis]
            dv = np.zeros(3); dv[v_axis] = 2*half[v_axis]
            nu = max(1, int(math.ceil(2*half[u_axis]/tile)))
            nv = max(1, int(math.ceil(2*half[v_axis]/tile)))
            t, p = _grid_quads(p0, du, dv, nu, nv)
            tris.append(t); parity.append(p)
    return np.concatenate(tris), np.concatenate(parity)


def _cylinder(radius, half, segments=16, caps=True):
    a = np.linspace(0, 2*np.pi, segments+1)
    ring = np.stack([radius*np.cos(a), radius*np.sin(a)], 1)
    tris = []
    for k in range(segments):
        p0 = [*ring[k], -half]; p1 = [*ring[k+1], -half]
        p2 = [*ring[k+1], half]; p3 = [*ring[k], half]
        tris += [(p0, p1, p2), (p0, p2, p3)]
        if caps:
            tris += [([0, 0, half], p3, p2), ([0, 0, -half], p1, p0)]
    return np.asarray(tris, float)


def _capsule(radius, half, segments=12):
    tris = [_cylinder(radius, half, segments, caps=False)]
    sphere = _icosphere(1)*radius
    top = sphere.copy(); top[..., 2] = np.where(top[..., 2] >= 0, top[..., 2]+half, top[..., 2]-half)
    tris.append(top)
    return np.concatenate(tris)


def _decimate(vertices, faces, cell):
    """Vertex-clustering simplification for sub-pixel CAD detail (render only)."""
    keys = np.floor(vertices/cell).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    count = np.bincount(inverse)
    merged = np.stack([np.bincount(inverse, weights=vertices[:, k]) for k in range(3)], 1)/count[:, None]
    f = inverse[faces]
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = f[keep]
    _, first = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
    return merged, f[np.sort(first)]


class ReplayGeometry:
    """Body-local triangles for every visible source geom, in manifest body order."""

    def __init__(self, manifest, scene_xml, *, tile=1.0, decimate_faces=4000, decimate_cell=.012):
        import mujoco
        self.model_path = str(Path(scene_xml).resolve())
        m = mujoco.MjModel.from_xml_path(self.model_path)
        names = [b['name'] for b in manifest['bodies']]
        lookup = {name: i for i, name in enumerate(names)}
        local, body_index, colors, checker = [], [], [], []
        self.geom_count = 0
        for g in range(m.ngeom):
            rgba = m.geom_rgba[g]
            if rgba[3] <= 0 or m.geom_group[g] >= 3:
                continue  # MuJoCo default-visible visual geometry only
            kind = int(m.geom_type[g]); size = m.geom_size[g]
            parity = None
            if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
                tris, parity = _box(size, tile)
            elif kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                tris = _icosphere(1)*size[0]
            elif kind == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                tris = _icosphere(1)*size[:3]
            elif kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                tris = _capsule(size[0], size[1])
            elif kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                tris = _cylinder(size[0], size[1])
            elif kind == int(mujoco.mjtGeom.mjGEOM_MESH):
                mesh = int(m.geom_dataid[g])
                va, vn = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
                fa, fn = int(m.mesh_faceadr[mesh]), int(m.mesh_facenum[mesh])
                vertices = np.asarray(m.mesh_vert[va:va+vn], float)
                faces = np.asarray(m.mesh_face[fa:fa+fn], np.int64)
                if fn > decimate_faces:
                    vertices, faces = _decimate(vertices, faces, decimate_cell)
                tris = vertices[faces]
            elif kind == int(mujoco.mjtGeom.mjGEOM_PLANE):
                half = [size[0] if size[0] > 0 else 25., size[1] if size[1] > 0 else 25., 0.]
                tris, parity = _grid_quads(np.array([-half[0], -half[1], 0.]), np.array([2*half[0], 0, 0]),
                                           np.array([0, 2*half[1], 0]), max(1, int(2*half[0]/tile)),
                                           max(1, int(2*half[1]/tile)))
            else:
                continue
            R = _quat_matrix(m.geom_quat[g]); p = np.asarray(m.geom_pos[g], float)
            tris = tris @ R.T + p
            body = int(m.geom_bodyid[g])
            index = -1 if body == 0 else lookup[m.body(body).name]
            n = len(tris)
            local.append(tris); body_index.append(np.full(n, index))
            colors.append(np.repeat(np.asarray(rgba[:3], float)[None], n, 0))
            checker.append(np.zeros(n) if parity is None or body != 0 else np.where(parity, -.035, .035))
            self.geom_count += 1
        self.local = np.concatenate(local)
        self.body_index = np.concatenate(body_index)
        self.colors = np.concatenate(colors)
        self.checker = np.concatenate(checker)
        self.triangle_count = len(self.local)
        self.body_count = len(names)


class Camera:
    def __init__(self, position, look_at, width, height, horizontal_fov_deg=47.2, near=.1):
        self.position = np.asarray(position, float)
        forward = np.asarray(look_at, float)-self.position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0., 0., 1.])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        self.rotation = np.stack([right, down, forward])
        self.width, self.height, self.near = int(width), int(height), float(near)
        self.focal = .5*width/math.tan(math.radians(horizontal_fov_deg)/2)
        self.cx, self.cy = .5*width, .5*height


def world_triangles(geometry, positions, quaternions_wxyz):
    positions = np.asarray(positions, float)
    rotations = _quat_matrix(quaternions_wxyz)
    if positions.shape != (geometry.body_count, 3) or rotations.shape != (geometry.body_count, 3, 3):
        raise ValueError('Pose arrays do not match the manifest body inventory')
    out = geometry.local.copy()
    moving = geometry.body_index >= 0
    idx = geometry.body_index[moving]
    out[moving] = np.einsum('tij,tvj->tvi', rotations[idx], geometry.local[moving])+positions[idx][:, None, :]
    return out


def _shade(tris, colors, checker, camera):
    normal = np.cross(tris[:, 1]-tris[:, 0], tris[:, 2]-tris[:, 0])
    length = np.linalg.norm(normal, axis=1)
    normal = normal/np.maximum(length, 1e-15)[:, None]
    toward = camera.position-tris.mean(axis=1)
    normal *= np.where(np.einsum('ij,ij->i', normal, toward) < 0, -1., 1.)[:, None]
    diffuse = np.clip(normal @ LIGHT, 0., 1.)
    headlight = np.clip(np.einsum('ij,ij->i', normal, toward/np.linalg.norm(toward, axis=1)[:, None]), 0., 1.)
    shade = .30+.55*diffuse+.20*headlight
    rgb = colors*shade[:, None]*(1.+checker[:, None])
    return np.clip(rgb, 0., 1.), length > 1e-14


def rasterize(tris_world, rgb, camera, max_batch=2_000_000):
    """Z-buffer rasterization with perspective-correct depth (1/z) comparisons."""
    W, H = camera.width, camera.height
    cam = (tris_world-camera.position) @ camera.rotation.T
    z = cam[..., 2]
    keep = np.all(z > camera.near, axis=1)
    cam, rgb = cam[keep], rgb[keep]
    z = cam[..., 2]
    u = camera.focal*cam[..., 0]/z+camera.cx
    v = camera.focal*cam[..., 1]/z+camera.cy
    x0 = np.floor(u.min(1)).astype(np.int64); x1 = np.ceil(u.max(1)).astype(np.int64)
    y0 = np.floor(v.min(1)).astype(np.int64); y1 = np.ceil(v.max(1)).astype(np.int64)
    visible = (x1 >= 0) & (x0 < W) & (y1 >= 0) & (y0 < H)
    area = (u[:, 1]-u[:, 0])*(v[:, 2]-v[:, 0])-(u[:, 2]-u[:, 0])*(v[:, 1]-v[:, 0])
    visible &= np.abs(area) > 1e-12
    u, v, z, rgb, area = u[visible], v[visible], z[visible], rgb[visible], area[visible]
    x0 = np.clip(x0[visible], 0, W-1); x1 = np.clip(x1[visible], 0, W-1)
    y0 = np.clip(y0[visible], 0, H-1); y1 = np.clip(y1[visible], 0, H-1)
    bw = x1-x0+1; bh = y1-y0+1; candidates = bw*bh
    inv_z = 1./z
    best = np.full(W*H, -np.inf)
    color = np.zeros((W*H, 3))
    bucket = np.ceil(np.log2(np.maximum(candidates, 1))).astype(np.int64)
    for b in np.unique(bucket):
        members = np.flatnonzero(bucket == b)
        size = int(2**b)
        step = max(1, max_batch//size)
        offsets = np.arange(size)
        for start in range(0, len(members), step):
            t = members[start:start+step]
            k = offsets[None, :]
            width = bw[t][:, None]
            px = x0[t][:, None]+k % width
            py = y0[t][:, None]+k//width
            valid = k < candidates[t][:, None]
            sx = px+.5; sy = py+.5
            ut, vt = u[t], v[t]
            a = area[t][:, None]
            w0 = ((ut[:, 1, None]-sx)*(vt[:, 2, None]-sy)-(ut[:, 2, None]-sx)*(vt[:, 1, None]-sy))/a
            w1 = ((ut[:, 2, None]-sx)*(vt[:, 0, None]-sy)-(ut[:, 0, None]-sx)*(vt[:, 2, None]-sy))/a
            w2 = 1.-w0-w1
            inside = valid & (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
            if not inside.any():
                continue
            iz = inv_z[t]
            depth = w0*iz[:, 0, None]+w1*iz[:, 1, None]+w2*iz[:, 2, None]
            rows, cols = np.nonzero(inside)
            pixel = py[rows, cols]*W+px[rows, cols]
            d = depth[rows, cols]
            tri = t[rows]
            order = np.lexsort((-d, pixel))
            pixel, d, tri = pixel[order], d[order], tri[order]
            first = np.ones(len(pixel), bool); first[1:] = pixel[1:] != pixel[:-1]
            pixel, d, tri = pixel[first], d[first], tri[first]
            closer = d > best[pixel]
            best[pixel[closer]] = d[closer]
            color[pixel[closer]] = rgb[tri[closer]]
    empty = ~np.isfinite(best)
    rows = np.repeat(np.linspace(0., 1., H), W)
    sky = SKY_TOP[None]*(1-rows[:, None])+SKY_BOTTOM[None]*rows[:, None]
    color[empty] = sky[empty]
    return (np.clip(color, 0., 1.)*255.+.5).astype(np.uint8).reshape(H, W, 3)


def stamp_text(image, text, x=8, y=8, scale=2, color=(20, 20, 20)):
    for n, char in enumerate(text):
        glyph = FONT_5X7.get(char)
        if glyph is None:
            continue
        for row, bits in enumerate(glyph):
            for col, bit in enumerate(bits):
                if bit == '1':
                    yy = y+row*scale; xx = x+(n*6+col)*scale
                    image[yy:yy+scale, xx:xx+scale] = color
    return image


def render_pose(geometry, camera, positions, quaternions_wxyz, label=None):
    tris = world_triangles(geometry, positions, quaternions_wxyz)
    rgb, ok = _shade(tris, geometry.colors, geometry.checker, camera)
    image = rasterize(tris[ok], rgb[ok], camera)
    if label:
        stamp_text(image, label)
    return image


def resolve_source_scene(result):
    """The MuJoCo scene named by the result's manifest, verified by its hash."""
    import hashlib
    result = Path(result)
    manifest = json.loads((result/'manifest.json').read_text(encoding='utf-8'))
    here = Path(__file__).resolve().parent
    candidates = []
    for base in (result, here/'generated'/str(manifest.get('case', 'soil_final'))):
        value = manifest.get('source_scene') or manifest.get('source_scene_xml')
        if value:
            candidates.append((base/value).resolve())
    for path in candidates:
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == manifest['source_scene_sha256']:
            return manifest, path
    raise FileNotFoundError('Original MuJoCo source scene with the manifest hash was not found: '
                            + ', '.join(map(str, candidates)))
