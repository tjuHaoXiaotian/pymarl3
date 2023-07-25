from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.maps import lib



class SMACMap(lib.Map):
    directory = "SMAC_Maps"
    download = "https://github.com/oxwhirl/smac#smac-maps"
    players = 2
    step_mul = 8
    game_steps_per_episode = 0


map_param_registry = {
    "10gen_terran": {
        "n_agents": 10,
        "n_enemies": 10,
        "limit": 400,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "terran_gen",
        "map_name": "32x32_flat",
    },
    "10gen_zerg": {
        "n_agents": 10,
        "n_enemies": 10,
        "limit": 400,
        "a_race": "Z",
        "b_race": "Z",
        "unit_type_bits": 3,
        "map_type": "zerg_gen",
        "map_name": "32x32_flat",
    },
    "10gen_protoss": {
        "n_agents": 10,
        "n_enemies": 10,
        "limit": 400,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 3,
        "map_type": "protoss_gen",
        "map_name": "32x32_flat",
    },
}


def get_smac_map_registry():
    return map_param_registry


for name, map_params in map_param_registry.items():
    globals()[name] = type(
        name, (SMACMap,), dict(filename=map_params["map_name"])
    )
