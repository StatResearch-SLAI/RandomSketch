#!/usr/bin/env bash
#SBATCH --job-name=randomsketch-study
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# Normally submitted by submit_all.sh, which supplies absolute paths and Slurm
# log destinations. No site-specific partition, account, or Python is assumed.
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    printf 'Usage: sbatch [site options] %s STUDY REPO_ROOT OUTPUT_ROOT [PYTHON]\n' "$0" >&2
    exit 2
fi

study="$1"
repo_root="$2"
output_root="$3"
case "$study" in
    conditional|accuracy|threshold|dense_benchmarks|isotropic_benchmarks) ;;
    *) printf 'Unknown study: %s\n' "$study" >&2; exit 2 ;;
esac

if [[ ! -f "$repo_root/reproduce.py" ]]; then
    printf 'Cannot find the reproduction runner: %s/reproduce.py\n' "$repo_root" >&2
    exit 2
fi
repo_root="$(cd -- "$repo_root" && pwd -P)"
mkdir -p -- "$output_root"
output_root="$(cd -- "$output_root" && pwd -P)"

if [[ $# -eq 4 ]]; then
    python_choice="$4"
elif [[ -n "${RANDOMSKETCH_PYTHON:-}" ]]; then
    python_choice="$RANDOMSKETCH_PYTHON"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    python_choice="$VIRTUAL_ENV/bin/python"
else
    python_choice=python3
fi
python_bin="$(command -v -- "$python_choice")" || {
    printf 'Python executable was not found: %s\n' "$python_choice" >&2
    exit 2
}
if [[ ! -x "$python_bin" ]]; then
    printf 'Python must name an executable file: %s\n' "$python_bin" >&2
    exit 2
fi
python_bin="$(cd -- "$(dirname -- "$python_bin")" && pwd -P)/$(basename -- "$python_bin")"

jobs="${SLURM_CPUS_PER_TASK:-6}"
if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Invalid SLURM_CPUS_PER_TASK: %s\n' "$jobs" >&2
    exit 2
fi
log_root="$output_root/slurm"
mkdir -p -- "$log_root"
job_key="${SLURM_JOB_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
log_prefix="$log_root/${study}-${job_key}"
job_metadata="${log_prefix}.job.txt"

# Keep one BLAS thread per Python worker. The study's own threadpool control
# applies the same policy after importing the numerical libraries.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${log_prefix}.matplotlib"
mkdir -p -- "$MPLCONFIGDIR"

# Dedicated logs also work when this runner is submitted directly with sbatch.
exec > >(tee "${log_prefix}.run.log") 2> >(tee "${log_prefix}.error.log" >&2)
finish() {
    local run_status=$?
    trap - EXIT
    printf 'finished_utc=%s\nexit_code=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$run_status" >> "$job_metadata"
    exit "$run_status"
}
trap finish EXIT

{
    printf 'started_utc=%s\nstudy=%s\nrepo_root=%s\noutput_root=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$study" "$repo_root" "$output_root"
    printf 'python=%s\nworkers=%s\nhost=%s\n' "$python_bin" "$jobs" "$(hostname)"
    printf 'slurm_job_id=%s\nslurm_job_name=%s\nslurm_partition=%s\nslurm_nodes=%s\n' \
        "${SLURM_JOB_ID:-}" "${SLURM_JOB_NAME:-}" "${SLURM_JOB_PARTITION:-}" "${SLURM_JOB_NODELIST:-}"
    printf 'slurm_cpus_per_task=%s\nslurm_mem_per_node=%s\n' \
        "${SLURM_CPUS_PER_TASK:-}" "${SLURM_MEM_PER_NODE:-}"
    "$python_bin" --version
    if command -v git >/dev/null 2>&1; then
        if revision="$(git -C "$repo_root" rev-parse HEAD 2>/dev/null)"; then
            printf 'git_commit=%s\n' "$revision"
        fi
    fi
    printf 'command='
    printf '%q ' "$python_bin" "$repo_root/reproduce.py" run --study "$study" \
        --output-root "$output_root" --jobs "$jobs"
    printf '\n'
} > "$job_metadata"

if "$python_bin" -m pip freeze > "${log_prefix}.pip-freeze.txt" 2> "${log_prefix}.pip-freeze.stderr.txt"; then
    printf 'pip_freeze=complete\n' >> "$job_metadata"
else
    printf 'pip_freeze=unavailable; see pip-freeze.stderr.txt and installed-packages.txt\n' >> "$job_metadata"
    printf 'pip freeze is unavailable; recording installed package versions through Python metadata.\n' >&2
    "$python_bin" - <<'PY' > "${log_prefix}.installed-packages.txt"
from importlib.metadata import distributions
for entry in sorted({f"{dist.metadata['Name']}=={dist.version}" for dist in distributions()}):
    print(entry)
PY
fi

cd -- "$repo_root"
printf 'Running %s with %s workers; output root: %s\n' "$study" "$jobs" "$output_root"
"$python_bin" "$repo_root/reproduce.py" run --study "$study" \
    --output-root "$output_root" --jobs "$jobs"
