"""Development utility only. Runtime reads the already packed arrays."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
from scipy.spatial import ConvexHull
import trimesh
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cmg_isaac.geometry import geometry_records

def main():
    rec=geometry_records(ROOT)
    data={}; info={}
    for r in rec:
        if 'mesh' not in r or r['mesh'] in info:
            continue
        f=ROOT/'upstream/franka_emika_panda/assets'/r['file']
        m=trimesh.load_mesh(f, process=False)
        if isinstance(m,trimesh.Scene):
            m=m.to_geometry()
        v=np.asarray(m.vertices,dtype=np.float64)*r['scale']
        faces=np.asarray(m.faces,dtype=np.int32)
        hull=None
        if r['collision']:
            # MuJoCo mesh contacts use the convex hull as well. This is not
            # visual-mesh substitution or an invented primitive collider.
            unique=np.unique(v,axis=0)
            h=ConvexHull(unique)
            remap={old:i for i,old in enumerate(h.vertices)}
            hull=len(h.vertices)
            v=unique[h.vertices]
            faces=np.asarray([[remap[int(k)] for k in face] for face in h.simplices],dtype=np.int32)
            # Orient the triangles outwards. PhysX recooks a convex hull.
            center=v.mean(axis=0)
            for face in faces:
                a,b,c=v[face]
                if np.dot(np.cross(b-a,c-a),(a+b+c)/3-center)<0:
                    face[1],face[2]=face[2],face[1]
        key=r['mesh']
        data[key+'_points']=v.astype(np.float32)
        data[key+'_faces']=faces
        info[key]=dict(source_file=r['file'],sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                       vertices=len(v),triangles=len(faces),collision_hull_vertices=hull,
                       bounds=[v.min(axis=0).tolist(),v.max(axis=0).tolist()])
    np.savez_compressed(ROOT/'data/mesh_arrays.npz',**data)
    (ROOT/'data/geometry.json').write_text(json.dumps(dict(geometries=rec,meshes=info),indent=2)+'\n')
    print(f'{len(info)} meshes, {len(rec)} source geometries, {sum(r["pad"] for r in rec)} finger pads')
    print('Largest collision hull:',max(i['collision_hull_vertices'] or 0 for i in info.values()))
if __name__=='__main__':main()
