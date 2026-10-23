# Learning to Reason with Selective Internal Feedback



## SIF-Q: continuous quantile comparison with peer voting

The new SIF-Q method replaces fixed certainty windows with an exact comparison
of empirical quantile functions across the full quantile interval. An integer
boundary merge computes pairwise net support without a sampled quantile grid.
All nonempty responses participate, with no response-length eligibility cutoff.

Version A retains binary peer voting (default peer win fraction: 0.75),
group-relative advantage normalization, and group-level filtering shared by the
actor and KL objectives. The training entrypoint remains
`algorithm.adv_estimator=intuitor`; the new method parameter is
`algorithm.peer_threshold=0.75`. Method effectiveness remains to be evaluated.

- [Method implementation and configuration](verl-intuitor/README.md)
- [Qwen2.5-Math-1.5B eight-GPU training script](running/train_sif_qwen25_math_1_5b.sh)
- [Exact comparison and integration tests](verl-intuitor/tests/trainer/ppo/test_vote_intuitor_on_cpu.py)

The new training script keeps the reference GRPO training settings and saves
periodic checkpoints and runs validation every 10 steps. Server paths can be
overridden using the environment variables defined in the script.

### Validation metric

With the current validation settings (`n=1`, `do_sample=False`, `temperature=0`),
the metric is
`val-core/DigitalLearningGmbH/MATH-lighteval/reward/mean@1`:
one greedy answer per prompt, scored 0/1 using this repository's boxed-answer
extraction and normalized answer matching. This is not a run of the full
Hugging Face Lighteval evaluation pipeline.

For the current MATH test parquet and Qwen2.5-Math-1.5B tokenizer, the inherited
512-token prompt limit retains 4,965 of 5,000 validation rows. This run therefore
reports accuracy on the retained subset, not the complete MATH-5000 set.
The prompt input limit is separate from the removed response-length filter.
