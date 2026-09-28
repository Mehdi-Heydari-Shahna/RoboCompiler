import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
from kangaroo_isaac.model import Model
from kangaroo_isaac.control import Reference
from kangaroo_isaac.cut_graph import CutGraph
from kangaroo_isaac.source_dynamics import SourceDynamics
import numpy as np

@pytest.fixture(scope='session')
def model():return Model()

@pytest.fixture(scope='session')
def ref():return Reference()

@pytest.fixture(scope='session')
def graph(model):return CutGraph(model.c,np.asarray(model.c['initial_seed']))

@pytest.fixture(scope='session')
def dynamics(model):return SourceDynamics(model.c,model.ids)
