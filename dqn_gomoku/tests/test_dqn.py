"""dqn_gomoku 包的单元测试。"""
import os
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from alpha_zero.envs.gomoku import GomokuEnv

from dqn_gomoku.agent import DQNAgent, Transition
from dqn_gomoku.evaluate import play_match
from dqn_gomoku.net import QNet, masked_argmax
from dqn_gomoku.opponents import heuristic_policy
from dqn_gomoku.selfplay import play_episode
from dqn_gomoku.symmetry import transform_action, transform_mask, transform_planes


class _ScriptedAgent:
    """按固定顺序出动作的假 agent，供测试 selfplay 回填逻辑使用。"""

    def __init__(self, actions):
        self._actions = list(actions)
        self._i = 0

    def select_action(self, obs, legal, epsilon):
        action = self._actions[self._i]
        self._i += 1
        return action


class TestSymmetry(unittest.TestCase):
    def setUp(self):
        self.board_size = 9
        rng = np.random.default_rng(0)
        self.planes = rng.integers(0, 2, size=(3, self.board_size, self.board_size)).astype(np.int8)
        self.action = int(rng.integers(0, self.board_size * self.board_size))
        self.mask = rng.integers(0, 2, size=(self.board_size * self.board_size,)).astype(np.int8)

    def test_identity(self):
        np.testing.assert_array_equal(transform_planes(self.planes, 0), self.planes)
        self.assertEqual(transform_action(self.action, self.board_size, 0), self.action)

    def test_action_matches_plane_transform(self):
        orig_r, orig_c = divmod(self.action, self.board_size)
        for sym in range(8):
            one_hot = np.zeros((1, self.board_size, self.board_size), dtype=np.int8)
            one_hot[0, orig_r, orig_c] = 1
            transformed_one_hot = transform_planes(one_hot, sym)

            new_action = transform_action(self.action, self.board_size, sym)
            r, c = divmod(new_action, self.board_size)

            self.assertEqual(transformed_one_hot.sum(), 1)
            self.assertEqual(transformed_one_hot[0, r, c], 1)

    def test_mask_matches_plane_transform(self):
        for sym in range(8):
            m1 = transform_mask(self.mask, self.board_size, sym)
            m2 = transform_planes(self.mask.reshape(1, self.board_size, self.board_size), sym).reshape(-1)
            np.testing.assert_array_equal(m1, m2)


class TestSelfplayBackfill(unittest.TestCase):
    def test_black_win_backfill(self):
        env = GomokuEnv(board_size=9, num_stack=1)
        # 黑：第0行第0..4列连成5子；白：第1行第0..3列，与黑交替落子。
        actions = [0, 9, 1, 10, 2, 11, 3, 12, 4]
        agent = _ScriptedAgent(actions)

        transitions, winner, steps = play_episode(env, agent, epsilon=0.0)

        self.assertEqual(winner, env.black_player)
        self.assertEqual(steps, 9)
        self.assertEqual(len(transitions), len(actions))

        for i, t in enumerate(transitions):
            self.assertEqual(t.action, actions[i])
            if i == len(actions) - 1:
                self.assertEqual(t.reward, 1.0)
                self.assertTrue(t.done)
            elif i == len(actions) - 2:
                self.assertEqual(t.reward, -1.0)
                self.assertTrue(t.done)
            else:
                self.assertEqual(t.reward, 0.0)
                self.assertFalse(t.done)

    def test_play_episode_with_opponent(self):
        """白方由 heuristic_policy 外部对手操作，转移记录方式应与纯自对弈完全一致。"""
        env = GomokuEnv(board_size=9, num_stack=1)
        agent = DQNAgent(board_size=9, in_planes=3, device='cpu', channels=4, num_layers=1)

        transitions, winner, steps = play_episode(
            env, agent, epsilon=1.0, opponent=heuristic_policy, opponent_color=env.white_player
        )

        self.assertEqual(len(transitions), steps)
        for t in transitions:
            self.assertGreaterEqual(t.action, 0)
            self.assertLessEqual(t.action, 80)
        self.assertTrue(env.is_game_over())


def _first_legal_policy(env):
    """恒定策略：总是选第一个合法动作。"""
    legal = np.flatnonzero(env.legal_actions == 1)
    return int(legal[0])


class TestRandomOpening(unittest.TestCase):
    def test_random_opening_varies(self):
        # random_opening=2 时，不同随机种子下前两手（整局走子序列）应该出现至少两种不同开局。
        openings = set()
        for seed in range(10):
            random.seed(seed)
            env = GomokuEnv(board_size=9, num_stack=1)
            play_match(env, _first_legal_policy, _first_legal_policy, random_opening=2)
            moves = tuple(pm.move for pm in env.history)
            openings.add(moves[:2])
        self.assertGreaterEqual(len(openings), 2)

        # random_opening=0 时，两个恒定策略的整局走子完全可复现，与随机种子无关。
        full_games = set()
        for seed in (1, 2, 3):
            random.seed(seed)
            env = GomokuEnv(board_size=9, num_stack=1)
            play_match(env, _first_legal_policy, _first_legal_policy, random_opening=0)
            moves = tuple(pm.move for pm in env.history)
            full_games.add(moves)
        self.assertEqual(len(full_games), 1)


class TestComputeTargets(unittest.TestCase):
    def test_negamax_sign(self):
        board_size = 5
        in_planes = 3
        agent = DQNAgent(board_size=board_size, in_planes=in_planes, device='cpu', channels=4, num_layers=1)

        def fake_target_forward(x):
            n = x.shape[0]
            return torch.full((n, board_size * board_size), 0.5)

        agent.target.forward = fake_target_forward

        rewards = np.array([1.0, -1.0, 0.2], dtype=np.float32)
        dones = np.array([True, False, False])
        next_states = np.zeros((3, in_planes, board_size, board_size), dtype=np.int8)
        next_legal = np.ones((3, board_size * board_size), dtype=np.int8)

        targets = agent.compute_targets(rewards, next_states, next_legal, dones)

        expected_non_terminal = -agent.gamma * 0.5
        self.assertAlmostEqual(targets[0].item(), 1.0, places=5)
        self.assertAlmostEqual(targets[1].item(), expected_non_terminal, places=5)
        self.assertAlmostEqual(targets[2].item(), expected_non_terminal, places=5)


class TestDoubleDQN(unittest.TestCase):
    def test_double_dqn_uses_online_to_select(self):
        """Double DQN：动作选择用 online 网络的 argmax，评估用 target 网络在该动作上的值，
        而不是 target 网络自己的 argmax（后者是普通 DQN 会过估计的写法）。"""
        board_size = 5
        in_planes = 3
        A = board_size * board_size
        agent = DQNAgent(board_size=board_size, in_planes=in_planes, device='cpu', channels=4, num_layers=1)

        # target 网络：每个动作的值各不相同，自身 argmax 落在最后一个动作（A-1）
        target_row = torch.arange(A).float() / A

        def fake_target_forward(x):
            n = x.shape[0]
            return target_row.unsqueeze(0).repeat(n, 1)

        # online 网络：合法动作中 argmax 落在动作 0
        online_row = torch.zeros(A)
        online_row[0] = 10.0

        def fake_online_forward(x):
            n = x.shape[0]
            return online_row.unsqueeze(0).repeat(n, 1)

        agent.target.forward = fake_target_forward
        agent.online.forward = fake_online_forward

        rewards = np.array([0.0], dtype=np.float32)
        dones = np.array([False])
        next_states = np.zeros((1, in_planes, board_size, board_size), dtype=np.int8)
        next_legal = np.ones((1, A), dtype=np.int8)

        targets = agent.compute_targets(rewards, next_states, next_legal, dones)

        expected = -agent.gamma * target_row[0].item()  # target 在 online 选出的动作 0 上的值
        wrong = -agent.gamma * target_row.max().item()  # 错误写法：target 自己的 argmax

        self.assertAlmostEqual(targets[0].item(), expected, places=5)
        self.assertNotAlmostEqual(targets[0].item(), wrong, places=5)


class TestUpdateAugmentationConsistency(unittest.TestCase):
    def test_update_augmentation_consistency(self):
        """update(augment=True) 内部对 state/action/next_state/next_legal 用同一个 sym 做变换，
        且 gather 动作时保持对齐：手动做完全相同的对称变换后关闭增强，loss 应与开启增强一致。"""
        board_size = 5
        in_planes = 3
        channels = 4
        num_layers = 1
        sym = 5

        # 从一局随机自对弈里取 4 条真实 transition
        env = GomokuEnv(board_size=board_size, num_stack=1)
        gen_agent = DQNAgent(board_size=board_size, in_planes=in_planes, device='cpu', channels=channels, num_layers=num_layers)
        transitions, _winner, _steps = play_episode(env, gen_agent, epsilon=1.0)
        batch_raw = transitions[:4]

        # 手动用 symmetry 模块把这 4 条的 state/action/next_state/next_legal 变换好
        batch_manual = []
        for t in batch_raw:
            batch_manual.append(
                Transition(
                    state=transform_planes(t.state, sym),
                    action=transform_action(t.action, board_size, sym),
                    reward=t.reward,
                    next_state=transform_planes(t.next_state, sym),
                    next_legal=transform_mask(t.next_legal, board_size, sym),
                    done=t.done,
                )
            )

        # 两个初始权重相同、各自独立的 agent
        torch.manual_seed(123)
        agent_a = DQNAgent(board_size=board_size, in_planes=in_planes, device='cpu', channels=channels, num_layers=num_layers)
        torch.manual_seed(123)
        agent_b = DQNAgent(board_size=board_size, in_planes=in_planes, device='cpu', channels=channels, num_layers=num_layers)

        with mock.patch('dqn_gomoku.agent.random.randint', return_value=sym):
            loss_a = agent_a.update(batch_manual, augment=False)
            loss_b = agent_b.update(batch_raw, augment=True)

        self.assertLess(abs(loss_a - loss_b), 1e-5)


class TestMaskedArgmax(unittest.TestCase):
    def test_illegal_excluded(self):
        q = torch.tensor([[5.0, 1.0, 2.0]])
        legal = torch.tensor([[0, 1, 1]])
        result = masked_argmax(q, legal)
        self.assertEqual(result.item(), 2)


class TestHeuristicPolicy(unittest.TestCase):
    def _build_black_four(self):
        env = GomokuEnv(board_size=9, num_stack=1)
        env.reset()
        env.step(0)  # 黑 (0,0)
        env.step(80)  # 白 (8,8)
        env.step(1)  # 黑 (0,1)
        env.step(79)  # 白 (8,7)
        env.step(2)  # 黑 (0,2)
        env.step(78)  # 白 (8,6)
        env.step(3)  # 黑 (0,3)，此时黑已连4，to_play 切换为白
        return env

    def test_black_completes_five(self):
        env = self._build_black_four()
        env.step(77)  # 白 (8,5)，不堵，轮到黑
        self.assertEqual(env.to_play, env.black_player)
        action = heuristic_policy(env)
        self.assertEqual(action, 4)  # (0,4)

    def test_white_blocks(self):
        env = self._build_black_four()
        self.assertEqual(env.to_play, env.white_player)
        action = heuristic_policy(env)
        self.assertEqual(action, 4)  # (0,4)


class TestQNetHeads(unittest.TestCase):
    def test_qnet_heads_output_shape(self):
        for head in ('conv', 'linear'):
            for board_size in (9, 15):
                net = QNet(in_planes=3, board_size=board_size, channels=4, num_layers=1, head=head)
                x = torch.zeros(2, 3, board_size, board_size)
                out = net(x)
                self.assertEqual(tuple(out.shape), (2, board_size * board_size))

    def test_conv_head_translation_equivariance(self):
        """conv 头是平移等变的全卷积结构：把输入图案整体平移后，Q 值图上对应位置的取值应保持不变。
        图案与两次放置位置都离边界至少 4 格，避免卷积的零填充边界效应干扰比较。"""
        board_size = 15
        in_planes = 3
        torch.manual_seed(0)
        net = QNet(in_planes, board_size, channels=8, num_layers=2, head='conv')
        net.eval()

        pattern = torch.rand(in_planes, 3, 3)

        r1, c1 = 6, 6
        dr, dc = 2, 3
        r2, c2 = r1 + dr, c1 + dc

        x1 = torch.zeros(in_planes, board_size, board_size)
        x1[:, r1 - 1:r1 + 2, c1 - 1:c1 + 2] = pattern
        x2 = torch.zeros(in_planes, board_size, board_size)
        x2[:, r2 - 1:r2 + 2, c2 - 1:c2 + 2] = pattern

        with torch.no_grad():
            q1 = net(x1.unsqueeze(0)).reshape(board_size, board_size)
            q2 = net(x2.unsqueeze(0)).reshape(board_size, board_size)

        win1 = q1[r1 - 1:r1 + 2, c1 - 1:c1 + 2]
        win2 = q2[r2 - 1:r2 + 2, c2 - 1:c2 + 2]

        torch.testing.assert_close(win1, win2, atol=1e-5, rtol=0)

    def test_old_checkpoint_loads_as_linear(self):
        """老 checkpoint 没有 'head' 键，from_checkpoint 应默认按 linear 加载以保持兼容。"""
        agent = DQNAgent(board_size=9, in_planes=3, device='cpu', head='linear')
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, 'ckpt.pt')
            agent.save(ckpt_path)

            payload = torch.load(ckpt_path, map_location='cpu')
            del payload['head']
            torch.save(payload, ckpt_path)

            loaded = DQNAgent.from_checkpoint(ckpt_path, device='cpu')
            self.assertEqual(loaded.head, 'linear')
            self.assertEqual(loaded.online.head_type, 'linear')


if __name__ == '__main__':
    unittest.main()
