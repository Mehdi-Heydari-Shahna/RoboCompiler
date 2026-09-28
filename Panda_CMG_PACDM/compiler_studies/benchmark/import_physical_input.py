"""Reproduce the one-time migration of supplied physical records into the frontend."""
import json,numpy as np
from src.compiler import from_original
from src.bootstrap import ROOT
from src.io_utils import save_json
cmg=json.loads((ROOT/'original/data/panda_cmg.json').read_text());q=np.load(ROOT/'original/data/reference.npz')['q'][0]
save_json(ROOT/'inputs/physical_graph_regenerated.json',from_original(cmg,q))
print('Regenerated physical records. Compare with inputs/physical_graph.json; no source data was changed.')
