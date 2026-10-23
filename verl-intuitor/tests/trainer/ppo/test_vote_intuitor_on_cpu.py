# Copyright 2026
# Licensed under the Apache License, Version 2.0

import unittest

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.core_algos import (
    AdvantageEstimator,
    compute_vote_intuitor_advantage,
    vote_intuitor_min_length,
    vote_intuitor_window_starts,
)


MIN_LENGTH = vote_intuitor_min_length(10, 50, 100)


def _constant_rollouts(values, length=MIN_LENGTH):
    self_certainties = torch.as_tensor(values, dtype=torch.float32).unsqueeze(-1).repeat(1, length)
    response_mask = torch.ones_like(self_certainties)
    return self_certainties, response_mask


class TestVoteIntuitor(unittest.TestCase):
    def test_min_length_is_stride_times_windows_plus_extra_half_window(self):
        self.assertEqual(MIN_LENGTH, 550)
        self.assertEqual(vote_intuitor_min_length(10, 50, 100), 50 * 10 + 50)
        self.assertEqual(vote_intuitor_window_starts(550, 10, 100), list(range(0, 500, 50)))
        self.assertEqual(vote_intuitor_window_starts(1000, 10, 100)[0], 0)
        self.assertEqual(vote_intuitor_window_starts(1000, 10, 100)[-1], 900)

    def test_first_window_is_top_tokens_and_last_window_is_lowest_tokens(self):
        length = 1000
        self_certainties = torch.zeros((2, length))
        # Ranked: [10]*100 + [9]*800 + [0]*100 vs [10]*100 + [9]*800 + [8]*100.
        # Window 0 is the top 100; window 9 is the lowest 100, not the middle.
        self_certainties[0, :100] = 10.0
        self_certainties[0, 100:900] = 9.0
        self_certainties[1, :100] = 10.0
        self_certainties[1, 100:900] = 9.0
        self_certainties[1, 900:] = 8.0
        response_mask = torch.ones_like(self_certainties)

        _, _, _, rewards, win_counts, active, eligible = compute_vote_intuitor_advantage(
            self_certainties=self_certainties,
            response_mask=response_mask,
            index=np.array(["prompt-0", "prompt-0"]),
            vote_number=10,
            least_vote=1,
            least_number=1,
        )

        self.assertTrue(eligible.all())
        self.assertEqual(vote_intuitor_window_starts(length, 10, 100)[-1], length - 100)
        # Only the lowest-100 window differs (0 vs 8), so rollout 1 wins.
        self.assertEqual(win_counts.tolist(), [0, 1])
        self.assertEqual(rewards.tolist(), [0.0, 1.0])
        self.assertTrue(active.all())

    def test_length_550_uses_overlapping_rank_windows(self):
        length = MIN_LENGTH
        self_certainties = torch.zeros((2, length))
        # After ranking: [10]*100 + [0]*450 vs [10]*150 + [0]*400.
        self_certainties[0, :100] = 10.0
        self_certainties[1, :150] = 10.0
        response_mask = torch.ones_like(self_certainties)

        _, _, _, rewards, win_counts, active, eligible = compute_vote_intuitor_advantage(
            self_certainties=self_certainties,
            response_mask=response_mask,
            index=np.array(["prompt-0", "prompt-0"]),
            vote_number=10,
            least_vote=2,
            least_number=1,
        )

        self.assertTrue(eligible.all())
        # Rollout 0 wins no window at least_vote=2; rollout 1 wins the overlapping middle windows.
        self.assertEqual(win_counts.tolist(), [0, 1])
        self.assertEqual(rewards.tolist(), [0.0, 1.0])
        self.assertTrue(active.all())

    def test_responses_shorter_than_550_are_not_updated(self):
        length = MIN_LENGTH
        self_certainties = torch.zeros((3, length))
        # Short but uniformly very confident: this used to win rank-bin votes.
        self_certainties[0] = 100.0
        self_certainties[1] = 2.0
        self_certainties[2] = 1.0
        response_mask = torch.ones_like(self_certainties)
        response_mask[0, MIN_LENGTH - 1 :] = 0

        advantages, _, _, rewards, win_counts, active, eligible = compute_vote_intuitor_advantage(
            self_certainties=self_certainties,
            response_mask=response_mask,
            index=np.array(["prompt-0"] * 3),
            vote_number=10,
            least_vote=8,
            least_number=1,
        )

        self.assertEqual(eligible.tolist(), [False, True, True])
        self.assertEqual(rewards.tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(win_counts[0].item(), 0)
        self.assertFalse(active[0])
        self.assertTrue(active[1])
        self.assertTrue(active[2])
        self.assertFalse(advantages[0].any())
        self.assertTrue(torch.all(advantages[1] > 0))
        self.assertTrue(torch.all(advantages[2] < 0))

    def test_default_threshold_rewards_rollouts_that_beat_at_least_six_peers(self):
        self_certainties, response_mask = _constant_rollouts(torch.arange(8.0, 0.0, -1.0))

        _, _, _, rewards, win_counts, active, eligible = compute_vote_intuitor_advantage(
            self_certainties=self_certainties,
            response_mask=response_mask,
            index=np.array(["prompt-0"] * 8),
        )

        self.assertTrue(eligible.all())
        self.assertEqual(win_counts.tolist(), [7, 6, 5, 4, 3, 2, 1, 0])
        self.assertEqual(rewards.tolist(), [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertTrue(active.all())

    def test_all_zero_reward_group_is_fully_masked_from_updates(self):
        self_certainties, response_mask = _constant_rollouts(torch.ones(8))

        advantages, returns, token_rewards, rewards, win_counts, active, eligible = compute_vote_intuitor_advantage(
            self_certainties=self_certainties,
            response_mask=response_mask,
            index=np.array(["prompt-0"] * 8),
        )

        self.assertTrue(eligible.all())
        self.assertFalse(rewards.any())
        self.assertFalse(win_counts.any())
        self.assertFalse(active.any())
        self.assertFalse(token_rewards.any())
        self.assertFalse(advantages.any())
        self.assertFalse(returns.any())

    def test_all_short_group_is_fully_masked_from_updates(self):
        self_certainties = torch.ones((8, MIN_LENGTH))
        response_mask = torch.ones_like(self_certainties)
        response_mask[:, MIN_LENGTH - 1 :] = 0

        advantages, _, _, rewards, win_counts, active, eligible = compute_vote_intuitor_advantage(
            self_certainties=self_certainties,
            response_mask=response_mask,
            index=np.array(["prompt-0"] * 8),
        )

        self.assertFalse(eligible.any())
        self.assertFalse(rewards.any())
        self.assertFalse(win_counts.any())
        self.assertFalse(active.any())
        self.assertFalse(advantages.any())

    def test_step_metrics_and_loss_mask_count_active_groups(self):
        from verl.trainer.ppo.ray_trainer import compute_advantage, compute_vote_intuitor_metrics

        ranked, ranked_mask = _constant_rollouts(torch.arange(8.0, 0.0, -1.0))
        tied, tied_mask = _constant_rollouts(torch.ones(8))
        self_certainties = torch.cat([ranked, tied])
        response_mask = torch.cat([ranked_mask, tied_mask])
        data = DataProto.from_dict(
            tensors={
                "self_certaintys": self_certainties,
                "response_mask": response_mask,
                "token_level_rewards": torch.zeros_like(self_certainties),
            },
            non_tensors={"uid": np.array(["active"] * 8 + ["inactive"] * 8)},
        )
        config = OmegaConf.create(
            {
                "vote_number": 10,
                "vote_stride": 50,
                "vote_window": 100,
                "least_vote": 8,
                "least_number": 6,
            }
        )

        data = compute_advantage(data, AdvantageEstimator.INTUITOR, config=config)
        metrics = compute_vote_intuitor_metrics(data)

        self.assertEqual(data.batch["token_level_rewards"].sum(dim=-1).tolist(), [1.0, 1.0] + [0.0] * 14)
        self.assertTrue(data.batch["actor_loss_mask"][:8].all())
        self.assertFalse(data.batch["actor_loss_mask"][8:].any())
        self.assertTrue(data.batch["intuitor_eligible"].all())
        self.assertEqual(metrics["vote_intuitor/updated_rollouts"], 8.0)
        self.assertEqual(metrics["vote_intuitor/updated_prompt_groups"], 1.0)
        self.assertEqual(metrics["vote_intuitor/all_zero_prompt_groups"], 1.0)
        self.assertEqual(metrics["vote_intuitor/too_short_rollouts"], 0.0)
        self.assertEqual(metrics["vote_intuitor/optimizer_step_skipped"], 0.0)

    def test_vote_threshold_validation(self):
        invalid_thresholds = [(0, 1, 1, 50, 100), (10, 0, 1, 50, 100), (10, 11, 1, 50, 100), (10, 7, 0, 50, 100)]
        for vote_number, least_vote, least_number, vote_stride, vote_window in invalid_thresholds:
            with self.subTest(
                vote_number=vote_number,
                least_vote=least_vote,
                least_number=least_number,
                vote_stride=vote_stride,
                vote_window=vote_window,
            ):
                with self.assertRaises(ValueError):
                    compute_vote_intuitor_advantage(
                        self_certainties=torch.ones((2, MIN_LENGTH)),
                        response_mask=torch.ones((2, MIN_LENGTH)),
                        index=np.array(["prompt-0", "prompt-0"]),
                        vote_number=vote_number,
                        least_vote=least_vote,
                        least_number=least_number,
                        vote_stride=vote_stride,
                        vote_window=vote_window,
                    )


if __name__ == "__main__":
    unittest.main()
