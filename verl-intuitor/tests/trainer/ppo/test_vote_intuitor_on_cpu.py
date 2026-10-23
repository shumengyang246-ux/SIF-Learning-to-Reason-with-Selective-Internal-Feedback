# Copyright 2026
# Licensed under the Apache License, Version 2.0

import math
import unittest
from fractions import Fraction

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss, compute_vote_intuitor_advantage
from verl.trainer.ppo.quantile_feedback import pairwise_quantile_support, quantile_net_support


def pack(profiles):
    width = max(map(len, profiles), default=0)
    values = torch.full((len(profiles), width), float("nan"))
    mask = torch.zeros_like(values)
    for row, profile in enumerate(profiles):
        values[row, : len(profile)] = torch.tensor(profile)
        mask[row, : len(profile)] = 1
    return values, mask


def feedback(profiles, uids=None, **kwargs):
    values, mask = pack(profiles)
    if uids is None:
        uids = ["prompt"] * len(profiles)
    return compute_vote_intuitor_advantage(values, mask, np.array(uids), **kwargs)


class TestExactQuantileSupport(unittest.TestCase):
    def test_document_examples(self):
        self.assertEqual(quantile_net_support([10, 6, 4, 2], [9, 9, 7, 7, 3, 3, 1, 1]), 0.5)
        self.assertEqual(quantile_net_support([1000] + [4] * 99, [6] * 20), -0.98)
        self.assertEqual(quantile_net_support([8, 5, 2], [8, 8, 5, 5, 2, 2]), 0)
        self.assertEqual(quantile_net_support([9, 3], [8, 6, 1]), 2 / 3)

    def test_exact_against_independent_fraction_partition(self):
        rng = np.random.default_rng(42)
        for li in range(1, 12):
            for lj in range(1, 12):
                a = sorted(rng.integers(-4, 5, li).tolist(), reverse=True)
                b = sorted(rng.integers(-4, 5, lj).tolist(), reverse=True)
                boundaries = sorted({Fraction(k, li) for k in range(li + 1)} | {Fraction(k, lj) for k in range(lj + 1)})
                expected = Fraction(0)
                for start, end in zip(boundaries, boundaries[1:]):
                    midpoint = (start + end) / 2
                    x, y = a[math.floor(midpoint * li)], b[math.floor(midpoint * lj)]
                    expected += (end - start) * (int(x > y) - int(x < y))
                self.assertEqual(quantile_net_support(a, b), float(expected))

    def test_matrix_properties(self):
        profiles = [[10, 6, 4, 2], [9, 9, 7, 7, 3, 3, 1, 1], [8, 5, 2, 0]]
        matrix = pairwise_quantile_support(profiles)
        np.testing.assert_array_equal(matrix, [[0, 0.5, 1], [-0.5, 0, 1], [-1, -1, 0]])
        np.testing.assert_array_equal(matrix, -matrix.T)
        self.assertTrue((np.abs(matrix) <= 1).all())

    def test_repetition_order_and_monotone_transform(self):
        a, b = [2, 10, 6, 4], [9, 7, 3, 1, 0]
        expected = quantile_net_support(a, b)
        self.assertEqual(quantile_net_support(a * 3, b * 2), expected)
        self.assertEqual(quantile_net_support(a[::-1], b[::-1]), expected)
        self.assertEqual(quantile_net_support([x**3 for x in a], [x**3 for x in b]), expected)

    def test_no_margin_tolerance(self):
        self.assertEqual(quantile_net_support([1.0 + 1e-12], [1.0]), 1)
        self.assertEqual(quantile_net_support([1.0], [1.0]), 0)

    def test_invalid_profiles(self):
        for profile in [[], [float("nan")], [float("inf")], [-float("inf")]]:
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                quantile_net_support(profile, [1])


class TestQuantilePeerVoting(unittest.TestCase):
    def test_single_token_and_mixed_lengths_all_participate(self):
        adv, ret, tokens, rewards, wins, active, valid = feedback([[100], [2] * 549, [1] * 600])
        self.assertTrue(valid.all())
        self.assertTrue(active.all())
        self.assertEqual(rewards.tolist(), [1, 0, 0])
        self.assertEqual(wins.tolist(), [2, 1, 0])
        self.assertTrue(adv[0, 0] > 0)
        self.assertFalse(adv[0, 1:].any())
        self.assertTrue((adv[1, :549] < 0).all())
        torch.testing.assert_close(adv, ret)
        torch.testing.assert_close(tokens.sum(-1), rewards)

    def test_document_A_normalizes_sample_std(self):
        profiles = [[10, 6, 4, 2], [9, 9, 7, 7, 3, 3, 1, 1], [8, 8, 5, 5, 2, 2, 0, 0]]
        adv, _, _, rewards, wins, active, _ = feedback(profiles)
        expected = (torch.tensor([1.0, 0.0, 0.0]) - 1 / 3) / (torch.tensor([1.0, 0.0, 0.0]).std() + 1e-6)
        torch.testing.assert_close(adv[:, 0], expected)
        self.assertEqual(rewards.tolist(), [1, 0, 0])
        self.assertEqual(wins.tolist(), [2, 1, 0])
        self.assertTrue(active.all())

    def test_eight_peers_preserve_default_binary_reward(self):
        result = feedback([[x] for x in range(8, 0, -1)])
        self.assertEqual(result[3].tolist(), [1, 1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(result[4].tolist(), list(range(7, -1, -1)))

    def test_denominator_uses_nonempty_peers_and_includes_ties(self):
        result = feedback([[9], [8], [7], [6], [5], [4], [3], []])
        self.assertEqual(result[3].tolist(), [1, 1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(result[6].tolist(), [True] * 7 + [False])
        self.assertFalse(result[0][-1].any())
        # Tied peers remain in the denominator: 2/3 is below 3/4.
        result = feedback([[3], [3], [2], [1]])
        self.assertEqual(result[4].tolist(), [2, 2, 1, 0])
        self.assertFalse(result[3].any())
        self.assertFalse(result[5].any())

    def test_empty_and_singleton_groups_skip_without_nan(self):
        for profiles in [[[], []], [[9], []], [[1]]]:
            with self.subTest(profiles=profiles):
                result = feedback(profiles)
                for tensor in result[:6]:
                    self.assertFalse(tensor.any())
                    self.assertTrue(torch.isfinite(tensor).all())

    def test_uniform_rewards_filter_including_preference_cycles(self):
        cycle = [[33, 21, 12], [32, 23, 11], [31, 22, 13]]
        for profiles, threshold, reward in [([[1], [1]], 0.75, 0), (cycle, 0.75, 0), (cycle, 0.5, 1)]:
            with self.subTest(threshold=threshold, profiles=profiles):
                result = feedback(profiles, peer_threshold=threshold)
                self.assertEqual(result[3].tolist(), [reward] * len(profiles))
                self.assertFalse(result[5].any())
                self.assertFalse(result[0].any())

    def test_groups_and_batch_reordering(self):
        profiles = [[5], [4], [100], [1], []]
        uids = ["a", "a", "b", "b", "a"]
        original = feedback(profiles, uids)
        permutation = [4, 2, 0, 3, 1]
        shuffled = feedback([profiles[i] for i in permutation], [uids[i] for i in permutation])
        for before, after in zip(original, shuffled):
            torch.testing.assert_close(before[permutation], after)
        self.assertEqual(original[3].tolist(), [1, 0, 1, 0, 0])

    def test_mask_holes_last_token_and_stop_gradient(self):
        values = torch.tensor([[99.0, 4.0, float("nan"), 3.0], [2.0, 88.0, 1.0, float("inf")]], requires_grad=True)
        mask = torch.tensor([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]])
        result = compute_vote_intuitor_advantage(values, mask, np.array(["p", "p"]))
        self.assertEqual(result[2][0].tolist(), [0, 0, 0, 1])
        self.assertFalse(result[0][~mask.bool()].any())
        self.assertFalse(result[0].requires_grad)
        self.assertFalse(result[2].requires_grad)

    def test_invalid_values_and_thresholds(self):
        for threshold in [0, -1, 1.1, float("nan"), float("inf")]:
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                feedback([[2], [1]], peer_threshold=threshold)
        for value in [float("nan"), float("inf")]:
            with self.assertRaisesRegex(ValueError, "finite"):
                feedback([[value], [1]])
        with self.assertRaisesRegex(ValueError, "zeros and ones"):
            compute_vote_intuitor_advantage(torch.ones(2, 1), torch.full((2, 1), 0.5), np.array([0, 0]))

    def test_optional_normalization_off(self):
        result = feedback([[3], [2], [1]], norm_adv_by_std_in_grpo=False)
        torch.testing.assert_close(result[0][:, 0], torch.tensor([2 / 3, -1 / 3, -1 / 3]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_matches_cpu(self):
        values, mask = pack([[9, 3], [8, 6, 1], [1]])
        uids = np.array([0, 0, 0])
        cpu = compute_vote_intuitor_advantage(values, mask, uids)
        gpu = compute_vote_intuitor_advantage(values.cuda(), mask.cuda(), uids)
        for expected, actual in zip(cpu, gpu):
            torch.testing.assert_close(expected, actual.cpu())


class TestTrainerIntegration(unittest.TestCase):
    def test_reward_mask_metrics_and_loss_gradients(self):
        from verl.trainer.ppo.ray_trainer import compute_advantage, compute_vote_intuitor_metrics

        values, mask = pack([[3], [2], [1], [5], [5], [], [7]])
        external_scores = torch.full_like(values, 123.0)
        data = DataProto.from_dict(
            tensors={
                "self_certaintys": values,
                "response_mask": mask,
                "token_level_scores": external_scores,
                "token_level_rewards": external_scores.clone(),
            },
            non_tensors={"uid": np.array(["active"] * 3 + ["tied"] * 2 + ["singleton"] * 2)},
        )
        data = compute_advantage(data, AdvantageEstimator.INTUITOR, config=OmegaConf.create({"peer_threshold": 0.75}))
        metrics = compute_vote_intuitor_metrics(data)
        self.assertEqual(data.batch["token_level_rewards"].sum(-1).tolist(), [1, 0, 0, 0, 0, 0, 0])
        torch.testing.assert_close(data.batch["token_level_scores"], external_scores)
        self.assertEqual(data.batch["actor_loss_mask"].flatten().tolist(), [1, 1, 1, 0, 0, 0, 0])
        self.assertEqual(metrics["vote_intuitor/updated_prompt_groups"], 1)
        self.assertEqual(metrics["vote_intuitor/uniform_reward_prompt_groups"], 1)
        self.assertEqual(metrics["vote_intuitor/insufficient_peer_prompt_groups"], 1)
        self.assertEqual(metrics["vote_intuitor/empty_rollouts"], 1)
        self.assertEqual(metrics["vote_intuitor/updated_response_tokens"], 3)
        self.assertEqual(metrics["vote_intuitor/optimizer_step_skipped"], 0)
        # The same mask used by dp_actor excludes inactive rows from PG and KL.
        pg = torch.ones_like(values, requires_grad=True)
        kl = torch.ones_like(values, requires_grad=True)
        loss_mask = data.batch["actor_loss_mask"]
        loss = agg_loss(pg, loss_mask, "token-mean") + agg_loss(kl, loss_mask, "token-mean")
        loss.backward()
        self.assertFalse(pg.grad[3:].any())
        self.assertFalse(kl.grad[3:].any())
        self.assertTrue(pg.grad[:3].all())
        self.assertTrue(kl.grad[:3].all())

    def test_hydra_config_accepts_one_token_responses(self):
        from pathlib import Path
        from types import SimpleNamespace

        from hydra import compose, initialize_config_dir

        from verl.trainer.config import AlgoConfig
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        config_dir = str(Path(__file__).resolve().parents[3] / "verl" / "trainer" / "config")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            config = compose(
                config_name="ppo_trainer",
                overrides=[
                    "algorithm.adv_estimator=intuitor",
                    "data.max_response_length=1",
                    "actor_rollout_ref.rollout.n=2",
                    "trainer.n_gpus_per_node=1",
                    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
                    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
                ],
            )
        self.assertEqual(config.algorithm.peer_threshold, AlgoConfig().peer_threshold)
        trainer = SimpleNamespace(config=config, use_critic=False, use_reference_policy=False)
        RayPPOTrainer._validate_config(trainer)
        for threshold in [0, 1.1, float("nan")]:
            config.algorithm.peer_threshold = threshold
            with self.assertRaises(ValueError):
                RayPPOTrainer._validate_config(trainer)
        config.algorithm.peer_threshold = 0.75
        config.actor_rollout_ref.rollout.n = 1
        with self.assertRaisesRegex(ValueError, "rollout.n"):
            RayPPOTrainer._validate_config(trainer)

    def test_all_filtered_step(self):
        from verl.trainer.ppo.ray_trainer import compute_advantage, compute_vote_intuitor_metrics

        values, mask = pack([[1], [1]])
        data = DataProto.from_dict(
            tensors={"self_certaintys": values, "response_mask": mask}, non_tensors={"uid": np.array([0, 0])}
        )
        data = compute_advantage(data, AdvantageEstimator.INTUITOR, config=OmegaConf.create({}))
        self.assertFalse(data.batch["intuitor_group_active"].any())
        self.assertEqual(compute_vote_intuitor_metrics(data)["vote_intuitor/optimizer_step_skipped"], 1)


if __name__ == "__main__":
    unittest.main()
