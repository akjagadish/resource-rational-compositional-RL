from gym.envs.registration import register

register(
    id='jagadish2022curriculum-v0',
    entry_point='envs.bandits:CompositionalBandit',
    kwargs={'max_steps_per_subtask': 5, 'num_actions': 6, 'curriculum': True, 'linear_first': True}
)

register(
    id='jagadish2022curriculum-v1',
    entry_point='envs.bandits:CompositionalBandit',
    kwargs={'max_steps_per_subtask': 5, 'num_actions': 6, 'curriculum': True, 'linear_first': False}
)

register(
    id='jagadish2022noncurriculum-v0',
    entry_point='envs.bandits:CompositionalBandit',
    kwargs={'max_steps_per_subtask': 5, 'num_actions': 6, 'curriculum': False, 'linear_first': True}
)

register(
    id='jagadish2022noncurriculum-v1',
    entry_point='envs.bandits:CompositionalBandit',
    kwargs={'max_steps_per_subtask': 5, 'num_actions': 6, 'curriculum': False, 'linear_first': False}
)

register(
    id='jagadish2022curriculum-v2',
    entry_point='envs.bandits:CompositionalBanditAlternative',
    kwargs={'max_steps_per_subtask': 5, 'num_actions': 6, 'curriculum': True, 'linear_first': True}
)

register(
    id='jagadish2022curriculum-v3',
    entry_point='envs.bandits:CompositionalBanditAlternative',
    kwargs={'max_steps_per_subtask': 5, 'num_actions': 6, 'curriculum': True, 'linear_first': False}
)

register(
    id='jagadish2022noncurriculum-v2',
    entry_point='envs.bandits:CompositionalBanditAlternative',
    kwargs={'max_steps_per_subtask': 5, 'num_actions': 6, 'curriculum': False, 'linear_first': True}
)


register(
    id='jagadish2022noncurriculum-v3',
    entry_point='envs.bandits:CompositionalBanditAlternative',
    kwargs={'max_steps_per_subtask': 5, 'num_actions': 6, 'curriculum': False, 'linear_first': False}
)
