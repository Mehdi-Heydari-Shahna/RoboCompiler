"""Stable USD paths shared by authoring and the native binding."""
ROBOT = '/World/Kangaroo'
BODIES = ROBOT + '/Bodies'
SCENE = '/physicsScene'
GROUND = '/World/Ground'

# One fixed, axis-aligned ground box; used by authoring and the support estimator.
GROUND_CENTER_M = (0., 0., -.05)
GROUND_SIZE_M = (20., 20., .1)
