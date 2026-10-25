#!/bin/bash
#SBATCH --job-name=sif-qwen25-math-1.5b
#SBATCH --account=project3
#SBATCH --partition=project3-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=380G
#SBATCH --time=UNLIMITED
#SBATCH --output=/data/home/zdhs0108/ysm/acl2027/SIF/training_log/sif_acl_qwen2.5_math_1.5b_log/%x-%j.out
#SBATCH --error=/data/home/zdhs0108/ysm/acl2027/SIF/training_log/sif_acl_qwen2.5_math_1.5b_log/%x-%j.err

# SIF-Q (GRPO, version A by default; USE_DIRECT_ADV=true enables B): Qwen2.5-Math-1.5B, exact quantile support + peer voting.
# Existing allocation: srun --jobid=44777 --overlap -N1 -n1 -c64 --gres=gpu:8 bash running/train_sif_qwen25_math_1_5b.sh
set -euo pipefail
set -x

# Slurm copies this script to /var/spool/slurmd, so do not infer the repo from BASH_SOURCE.
INTUITOR_ROOT="${INTUITOR_ROOT:-/data/home/zdhs0108/ysm/acl2027/SIF}"
if [[ ! -d "${INTUITOR_ROOT}/verl-intuitor" ]]; then
    echo "[error] verl-intuitor not found under INTUITOR_ROOT=${INTUITOR_ROOT}"
    exit 1
fi
ENV_DIR="${ENV_DIR:-/data/home/zdhs0108/.conda/envs/intuitor}"
VERL_ROOT="${VERL_ROOT:-${INTUITOR_ROOT}/verl-intuitor}"

# Use the requested environment directly; noninteractive Slurm shells may lack conda.
export PATH="${ENV_DIR}/bin:${PATH}"
export CONDA_PREFIX="${ENV_DIR}"
export CONDA_DEFAULT_ENV=intuitor
[[ -x "${ENV_DIR}/bin/python3" ]] || { echo "[error] environment missing: ${ENV_DIR}"; exit 1; }
cd "${VERL_ROOT}"

export PYTHONPATH="${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

_detect_host_ip() {
    local ifc ip_addr
    for ifc in "${NCCL_SOCKET_IFNAME:-}" bond4 eth0; do
        [[ -z "${ifc}" ]] && continue
        ip_addr="$(ip -4 addr show "${ifc}" 2>/dev/null | awk '/inet / {print $2}' | cut -d/ -f1 | head -1)"
        if [[ -n "${ip_addr}" ]]; then
            echo "${ip_addr}"
            return 0
        fi
    done
}
unset MY_HOST_IP RAY_NODE_IP_ADDRESS
HOST_IP="$(_detect_host_ip)"
if [[ -z "${HOST_IP}" ]]; then
    HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
fi
if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
    if ip link show bond4 >/dev/null 2>&1; then
        NCCL_SOCKET_IFNAME=bond4
    else
        NCCL_SOCKET_IFNAME=eth0
    fi
fi
export RAY_NODE_IP_ADDRESS="${HOST_IP}"
export MY_HOST_IP="${HOST_IP}"
export NCCL_SOCKET_IFNAME
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME}}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
echo "[info] ray_node_ip=${RAY_NODE_IP_ADDRESS} nccl_if=${NCCL_SOCKET_IFNAME} nccl_p2p_disable=${NCCL_P2P_DISABLE}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VERL_VLLM_DISTRIBUTED_BACKEND="${VERL_VLLM_DISTRIBUTED_BACKEND:-ray}"
unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
unset PYTORCH_CUDA_ALLOC_CONF
export HYDRA_FULL_ERROR=1
export ACCELERATE_LOG_LEVEL=info
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TORCHINDUCTOR_DISABLE="${TORCHINDUCTOR_DISABLE:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR:-/tmp}/triton_cache_local_$$}"
mkdir -p "${TRITON_CACHE_DIR}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

export MODEL_PATH="${MODEL_PATH:-/data/home/zdhs0108/ysm/models/Qwen2.5-Math-1.5B}"
export MATH_DATA_DIR="${MATH_DATA_DIR:-/data/home/zdhs0108/ysm/data/math}"
export TRAIN_FILES="${TRAIN_FILES:-${MATH_DATA_DIR}/train.parquet}"
export VAL_FILES="${VAL_FILES:-${MATH_DATA_DIR}/test.parquet}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/home/zdhs0108/ysm/acl2027/SIF/checkpoint}"
export PROJECT_NAME="${PROJECT_NAME:-verl}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-sif_acl_qwen2.5_math_1.5b}"
export TRAINING_SEED="${TRAINING_SEED:-42}"
if [[ ! "${TRAINING_SEED}" =~ ^[0-9]+$ ]] || (( 10#${TRAINING_SEED} > 4294967295 )); then
    echo "[error] TRAINING_SEED must be an integer in [0, 4294967295], got: ${TRAINING_SEED}"
    exit 1
fi
export PYTHONHASHSEED="${TRAINING_SEED}"
if [[ -z "${DEFAULT_LOCAL_DIR:-}" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        DEFAULT_LOCAL_DIR="${CHECKPOINT_ROOT}/${EXPERIMENT_NAME}_seed${TRAINING_SEED}_${SLURM_JOB_ID}"
    else
        DEFAULT_LOCAL_DIR="${CHECKPOINT_ROOT}/${EXPERIMENT_NAME}_seed${TRAINING_SEED}_$(date '+%Y%m%d_%H%M%S')"
    fi
fi
export DEFAULT_LOCAL_DIR
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-${INTUITOR_ROOT}/training_log/sif_acl_qwen2.5_math_1.5b_log}"
_log_model_name() {
    basename "${MODEL_PATH}" | sed -E 's/[^a-zA-Z0-9._-]+/_/g'
}
JOB_LOG_DIR="${TRAIN_LOG_DIR}"
mkdir -p "${JOB_LOG_DIR}" "${CHECKPOINT_ROOT}" "${DEFAULT_LOCAL_DIR}"
# Prevent a second copy of this experiment from starting in the same allocation.
exec 9>"${JOB_LOG_DIR}/train-${SLURM_JOB_ID:-local}.lock"
flock -n 9 || { echo "[error] this experiment is already running"; exit 1; }
export TRAIN_LOG="${TRAIN_LOG:-${JOB_LOG_DIR}/train-${SLURM_JOB_ID:-local}-$(date '+%Y%m%d_%H%M%S').log}"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "[error] Model not found: ${MODEL_PATH}"
    exit 1
fi
if [[ ! -f "${TRAIN_FILES}" ]]; then
    echo "[error] Train data not found: ${TRAIN_FILES}"
    exit 1
fi
if [[ ! -f "${VAL_FILES}" ]]; then
    echo "[error] Val data not found: ${VAL_FILES}"
    exit 1
fi

# Keep the reference training parameters; only replace SIF-specific feedback.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-4}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-3072}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export ROLLOUT_TP="${ROLLOUT_TP:-1}"
export USE_DIRECT_ADV="${USE_DIRECT_ADV:-false}"
export PEER_THRESHOLD="${PEER_THRESHOLD:-0.75}"
export GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.6}"
export LEARNING_RATE="${LEARNING_RATE:-3e-6}"
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.005}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export TEST_FREQ="${TEST_FREQ:-10}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export ENABLE_WANDB="${ENABLE_WANDB:-0}"
# periodic: keep global_step_*; best_and_latest: only best/ + latest/
export SAVE_MODE="${SAVE_MODE:-periodic}"
export BEST_CKPT_METRIC="${BEST_CKPT_METRIC:-}"
export RESUME_MODE="${RESUME_MODE:-auto}"
export RETRAIN="${RETRAIN:-1}"
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-64}}"

IFS=',' read -r -a __gpu_list <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#__gpu_list[@]}"
if (( NUM_GPUS != 8 )); then
    echo "[error] expected 8 allocated GPUs, got ${NUM_GPUS}"
    exit 1
fi

if [[ "${ENABLE_WANDB}" == "1" ]]; then
    LOGGER_CFG='["console","wandb"]'
else
    LOGGER_CFG='["console"]'
    unset WANDB_API_KEY || true
fi

BEST_CKPT_METRIC_ARG='null'
if [[ -n "${BEST_CKPT_METRIC}" ]]; then
    BEST_CKPT_METRIC_ARG="${BEST_CKPT_METRIC}"
fi

if [[ "${RETRAIN}" == "1" ]]; then
    RESUME_MODE="disable"
fi
export RESUME_MODE

echo "[info] env=${ENV_DIR}"
echo "[info] verl_root=${VERL_ROOT}"
echo "[info] model=${MODEL_PATH}"
echo "[info] train_files=${TRAIN_FILES}"
echo "[info] val_files=${VAL_FILES}"
echo "[info] checkpoint_dir=${DEFAULT_LOCAL_DIR} save_mode=${SAVE_MODE} save_freq=${SAVE_FREQ} test_freq=${TEST_FREQ}"
echo "[info] experiment=${EXPERIMENT_NAME}"
echo "[info] resume_mode=${RESUME_MODE} retrain=${RETRAIN} epochs=${TOTAL_EPOCHS}"
echo "[info] gpus=${NUM_GPUS} cuda_visible=${CUDA_VISIBLE_DEVICES} batch=${TRAIN_BATCH_SIZE} n=${ROLLOUT_N} tp=${ROLLOUT_TP}"
echo "[info] training_seed=${TRAINING_SEED} pythonhashseed=${PYTHONHASHSEED}"
echo "[info] SIF-Q exact quantiles; use_direct_adv=${USE_DIRECT_ADV}; peer_threshold(A only)=${PEER_THRESHOLD}; no response-length filtering"
echo "[info] train_log=${TRAIN_LOG}"

unset RAY_ADDRESS RAY_DASHBOARD_ADDRESS RAY_NAMESPACE RAY_RUNTIME_ENV
export RAY_ADDRESS=""
export RAY_DASHBOARD_ADDRESS=""
export RAY_ENABLE_DASHBOARD=0
# Do not stop unrelated Ray processes; main_ppo starts its own local Ray runtime.

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=intuitor \
    algorithm.use_direct_adv="${USE_DIRECT_ADV}" \
    algorithm.peer_threshold="${PEER_THRESHOLD}" \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.seed="${TRAINING_SEED}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.seed="${TRAINING_SEED}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTIL}" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.logger="${LOGGER_CFG}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${DEFAULT_LOCAL_DIR}" \
    trainer.resume_mode="${RESUME_MODE}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.save_mode="${SAVE_MODE}" \
    trainer.best_ckpt_metric="${BEST_CKPT_METRIC_ARG}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    ray_init.num_cpus="${RAY_NUM_CPUS}" \
    "$@" \
    2>&1 | tee "${TRAIN_LOG}"
