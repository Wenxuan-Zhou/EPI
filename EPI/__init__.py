from gym.envs.mujoco.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from gym.envs.registration import registry, register, make, spec

register(
    id='HopperOriginal-v0',
    entry_point='EPI.envs:HopperEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='HopperAvg-v0',
    entry_point='EPI.envs:HopperAvgEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='HopperHistory-v0',
    entry_point='EPI.envs:HopperHistoryEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='HopperDirect-v0',
    entry_point='EPI.envs:HopperDirectEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)


register(
    id='HopperLSTM-v0',
    entry_point='EPI.envs:HopperAvgEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='HopperOracle-v0',
    entry_point='EPI.envs:HopperOracleEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='HopperTask-v0',
    entry_point='EPI.envs:HopperTaskEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
    kwargs={"reset": False}
)

register(
    id='HopperTaskReset-v0',
    entry_point='EPI.envs:HopperTaskEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
    kwargs={"reset": True}
)

register(
    id='HopperInteraction-v0',
    entry_point='EPI.envs:HopperInteractionEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='StrikerOriginal-v0',
    entry_point='EPI.envs:StrikerEnv',
    max_episode_steps=200,
    reward_threshold=0,
)

register(
    id='StrikerAvg-v0',
    entry_point='EPI.envs:StrikerAvgEnv',
    max_episode_steps=200,
    reward_threshold=0,
)

register(
    id='StrikerLSTM-v0',
    entry_point='EPI.envs:StrikerAvgEnv',
    max_episode_steps=200,
    reward_threshold=0,
)

register(
    id='StrikerOracle-v0',
    entry_point='EPI.envs:StrikerOracleEnv',
    max_episode_steps=200,
    reward_threshold=0,
)

register(
    id='StrikerHistory-v0',
    entry_point='EPI.envs:StrikerHistoryEnv',
    max_episode_steps=200,
    reward_threshold=0,
)

register(
    id='StrikerDirect-v0',
    entry_point='EPI.envs:StrikerDirectEnv',
    max_episode_steps=200,
    reward_threshold=0,
)

register(
    id='StrikerTask-v0',
    entry_point='EPI.envs:StrikerTaskEnv',
    max_episode_steps=200,
    reward_threshold=0,
    kwargs={"reset": False}
)

register(
    id='StrikerTaskReset-v0',
    entry_point='EPI.envs:StrikerTaskEnv',
    max_episode_steps=200,
    reward_threshold=0,
    kwargs={"reset": True}
)

register(
    id='StrikerInteraction-v0',
    entry_point='EPI.envs:StrikerInteractionEnv',
    max_episode_steps=200,
    reward_threshold=0,
)


def init(env,
         default_reward_scale=1,
         prediction_reward_scale=0.5,
         num_of_envs=25,
         num_of_params=2,
         embedding_dimension=8,
         loss_type='interaction_separation'
         ):
    global DEFAULT_REWARD_SCALE
    global PREDICTION_REWARD_SCALE
    global NUM_OF_ENVS
    global NUM_OF_PARAMS
    global EMBEDDING_DIMENSION
    global ENV
    global LOSS_TYPE

    ENV = env
    DEFAULT_REWARD_SCALE = default_reward_scale
    LOSS_TYPE = loss_type
    PREDICTION_REWARD_SCALE = prediction_reward_scale

    if env == 'striker':
        NUM_OF_ENVS = num_of_envs
        NUM_OF_PARAMS = num_of_params
        EMBEDDING_DIMENSION = embedding_dimension
    elif env == 'hopper':
        NUM_OF_ENVS = num_of_envs
        NUM_OF_PARAMS = num_of_params
        EMBEDDING_DIMENSION = embedding_dimension
    else:
        print('Environment '+env+' does not exist.')