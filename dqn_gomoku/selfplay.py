"""自对弈生成训练数据：双方都用同一个 agent 的 ε-greedy 策略，也可指定外部对手。"""
from typing import Callable, List, Optional, Tuple

from .agent import DQNAgent, Transition


def play_episode(
    env,
    agent: DQNAgent,
    epsilon: float,
    opponent: Optional[Callable[..., int]] = None,
    opponent_color: Optional[int] = None,
) -> Tuple[List[Transition], Optional[int], int]:
    """跑一局对弈，agent 用 ε-greedy 策略，可选地让指定颜色由外部 opponent 操作。

    Args:
        opponent: act(env) -> int 的可调用对象；为 None 时退化为纯自对弈。
        opponent_color: 1（黑）或 2（白），指定由 opponent 操作的颜色；
            当 opponent is not None 且 env.to_play == opponent_color 时，
            该步动作由 opponent(env) 给出，否则仍由 agent.select_action(obs, legal, epsilon) 给出。

    转移记录方式完全不变：无论哪一方走的，都记录 (state, action, next_state, next_legal)，
    结束后同样回填奖励。DQN 是 off-policy 算法，对手的动作也是合法的训练样本，
    这样能让网络见到"堵四/冲四"这类自对弈里探索不到的动作及其后果。

    Returns:
        (transitions, winner, steps)：transitions 是回填好奖励和 done 的转移列表，
        winner 是 env.winner（1/2/None），steps 是 env.steps。
    """
    obs = env.reset()
    raw_steps = []  # 先记录 (state, action, next_state, next_legal)，奖励结束后统一回填
    done = False
    last_reward = 0.0
    while not done:
        legal = env.legal_actions
        if opponent is not None and env.to_play == opponent_color:
            action = opponent(env)
        else:
            action = agent.select_action(obs, legal, epsilon)
        next_obs, reward, done, _ = env.step(action)
        raw_steps.append((obs, action, next_obs, env.legal_actions.copy()))
        last_reward = reward
        obs = next_obs

    winner = env.winner
    steps = env.steps
    total = len(raw_steps)

    transitions: List[Transition] = []
    for i, (state, action, next_state, next_legal) in enumerate(raw_steps):
        if i == total - 1:
            # 最后一步：用环境返回的奖励（获胜 1.0，平局 0.0），并标记为终局。
            reward = last_reward
            step_done = True
        elif winner is not None and total >= 2 and i == total - 2:
            # env 的 reward 只在"刚落子并获胜"的那一步给 1.0，输家永远拿不到直接反馈。
            # 若不显式处理，负号目标（negamax）会把输家倒数第二步当作非终局样本，
            # 用 -gamma * Q(下一状态) 去 bootstrap，而下一状态正是对手即将获胜的局面，
            # 这样学到的信号会被严重滞后和稀释。所以直接把输家的最后一步标为终局、
            # reward=-1.0，让它立刻学到"这步导致了失败"。
            reward = -1.0
            step_done = True
        else:
            reward = 0.0
            step_done = False
        transitions.append(
            Transition(state=state, action=action, reward=reward, next_state=next_state, next_legal=next_legal, done=step_done)
        )

    return transitions, winner, steps
