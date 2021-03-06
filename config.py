
class Config:
    env: str = None
    gamma: float = None
    learning_rate: float = None
    learning_starts: int = None
    frames: int = None
    episodes: int = None

    max_buff: int = None
    prioritized_replay: bool = False
    prioritized_replay_alpha: float = 0.6
    prioritized_replay_beta0: float = 0.4
    prioritized_replay_beta_iters: int = None
    prioritized_replay_eps: float = 1e-6

    batch_size: int = None

    epsilon: float = None
    eps_fraction: float = None
    epsilon_min: float = None

    state_dim: int = None
    state_shape = None
    state_high = None
    state_low = None
    seed = None
    output = 'out'

    action_dim: int = None
    action_high = None
    action_low = None
    action_lim = None

    use_cuda: bool = None

    checkpoint: bool = False
    checkpoint_interval: int = None

    record: bool = False
    record_ep_interval: int = None

    log_interval: int = None
    print_interval: int = None

    update_tar_interval: int = None

    win_reward: float = None
    win_break: bool = None





