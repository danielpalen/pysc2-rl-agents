from collections import namedtuple

import numpy as np

from pysc2.lib import actions
from pysc2.lib import features


FlatFeature = namedtuple('FlatFeatures', ['index', 'type', 'scale', 'name'])

NUM_FUNCTIONS = len(actions.FUNCTIONS)
NUM_PLAYERS = features.SCREEN_FEATURES.player_id.scale

FLAT_FEATURES = [
    FlatFeature(0,  features.FeatureType.SCALAR, 1, 'player_id'),
    FlatFeature(1,  features.FeatureType.SCALAR, 1, 'minerals'),
    FlatFeature(2,  features.FeatureType.SCALAR, 1, 'vespene'),
    FlatFeature(3,  features.FeatureType.SCALAR, 1, 'food_used'),
    FlatFeature(4,  features.FeatureType.SCALAR, 1, 'food_cap'),
    FlatFeature(5,  features.FeatureType.SCALAR, 1, 'food_army'),
    FlatFeature(6,  features.FeatureType.SCALAR, 1, 'food_workers'),
    FlatFeature(7,  features.FeatureType.SCALAR, 1, 'idle_worker_count'),
    FlatFeature(8,  features.FeatureType.SCALAR, 1, 'army_count'),
    FlatFeature(9,  features.FeatureType.SCALAR, 1, 'warp_gate_count'),
    FlatFeature(10, features.FeatureType.SCALAR, 1, 'larva_count'),
]

is_spatial_action = {}
for name, arg_type in actions.TYPES._asdict().items():
    # HACK: we should infer the point type automatically
    is_spatial_action[arg_type] = name in ['minimap', 'screen', 'screen2']


def stack_ndarray_dicts(lst, axis=0):
    """Concatenate ndarray values from list of dicts
    along new axis."""
    res = {}
    for k in lst[0].keys():
        res[k] = np.stack([d[k] for d in lst], axis=axis)
    return res


class Preprocessor():
    """Compute network inputs from pysc2 observations.

    See https://github.com/deepmind/pysc2/blob/master/docs/environment.md
    for the semantics of the available observations.
    """

    def __init__(self, obs_spec):
        screen_channels  = len(features.SCREEN_FEATURES)
        minimap_channels = len(features.MINIMAP_FEATURES)
        flat_channels    = len(FLAT_FEATURES)
        available_actions_channels = NUM_FUNCTIONS

        def get_input_channels():
            """Get static channel dimensions of network inputs."""
            return {
                'screen' : screen_channels,
                'minimap': minimap_channels,
                'flat'   : flat_channels,
                'available_actions': available_actions_channels
            }

        def preprocess_obs(obs_list):
            return stack_ndarray_dicts([_preprocess_obs(o.observation) for o in obs_list])

        def _preprocess_spatial(spatial):
            return np.transpose(spatial, [1, 2, 0])

        def _preprocess_obs(obs):
            """
            Compute screen, minimap and flat network inputs from raw observations.
            """
            available_actions = np.zeros(NUM_FUNCTIONS, dtype=np.float32)
            available_actions[obs['available_actions']] = 1

            screen  = _preprocess_spatial(obs['screen'])
            minimap = _preprocess_spatial(obs['minimap'])

            flat = np.concatenate([obs['player']])
            # TODO available_actions, control groups, cargo, multi select, build queue

            return {
                'screen' : screen,
                'minimap': minimap,
                'flat'   : flat,
                'available_actions': available_actions
            }

        self.get_input_channels = get_input_channels
        self.preprocess_obs = preprocess_obs
