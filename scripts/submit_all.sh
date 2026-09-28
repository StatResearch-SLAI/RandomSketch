#!/usr/bin/env bash
# Submit five independent Slurm jobs using the user's selected Python environment.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "$script_dir/.." && pwd -P)"
output_root=""
partition="${RANDOMSKETCH_PARTITION:-}"
account="${RANDOMSKETCH_ACCOUNT:-}"
cpus="${RANDOMSKETCH_CPUS:-6}"
memory="${RANDOMSKETCH_MEMORY:-16G}"
walltime="${RANDOMSKETCH_TIME:-04:00:00}"
python_choice="${RANDOMSKETCH_PYTHON:-}"
extra_sbatch=()

usage() {
    cat <<'USAGE'
Usage: scripts/submit_all.sh [options] [-- additional sbatch options]

Submit conditional, accuracy, threshold, dense_benchmarks, and
isotropic_benchmarks as five separate jobs. Each uses the paper seed and
3,000 repetitions configured by reproduce.py (plus 20 frozen fits where used).

Options:
  --output-root DIR  Fresh output directory; default: REPO/runs/slurm-UTCSTAMP
  --partition NAME   Site partition; default: RANDOMSKETCH_PARTITION or Slurm default
  --account NAME     Site account; default: RANDOMSKETCH_ACCOUNT or Slurm default
  --cpus N          CPUs/workers per study; default: RANDOMSKETCH_CPUS or 6
  --mem SIZE        Memory per study; default: RANDOMSKETCH_MEMORY or 16G
  --time TIME       Wall time per study; default: RANDOMSKETCH_TIME or 04:00:00
  --python PATH     Python executable; default: RANDOMSKETCH_PYTHON, active venv,
                    or python3 on PATH, in that order
  -h, --help        Show this message

The selected environment must already contain the runtime dependencies and
be accessible on compute nodes. Slurm inherits the submission environment.
Additional sbatch options after -- support site settings such as --qos.
Logs, pip freeze, job metadata, and the submission manifest go to DIR/slurm/.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-root|--partition|--account|--cpus|--mem|--time|--python)
            if [[ $# -lt 2 || -z "$2" ]]; then
                printf 'Missing value for %s\n' "$1" >&2
                exit 2
            fi
            case "$1" in
                --output-root) output_root="$2" ;;
                --partition) partition="$2" ;;
                --account) account="$2" ;;
                --cpus) cpus="$2" ;;
                --mem) memory="$2" ;;
                --time) walltime="$2" ;;
                --python) python_choice="$2" ;;
            esac
            shift 2 ;;
        --) shift; extra_sbatch=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! "$cpus" =~ ^[1-9][0-9]*$ ]]; then
    printf 'CPUs must be a positive integer: %s\n' "$cpus" >&2
    exit 2
fi
command -v sbatch >/dev/null 2>&1 || {
    printf 'sbatch is not on PATH. Run this script on a Slurm submission host.\n' >&2
    exit 2
}
if [[ ! -f "$repo_root/reproduce.py" ]]; then
    printf 'Cannot find the reproduction runner: %s/reproduce.py\n' "$repo_root" >&2
    exit 2
fi
if [[ -z "$python_choice" ]]; then
    if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
        python_choice="$VIRTUAL_ENV/bin/python"
    else
        python_choice=python3
    fi
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

if [[ -z "$output_root" ]]; then
    output_root="$repo_root/runs/slurm-$(date -u +%Y%m%dT%H%M%SZ)-$$"
fi
mkdir -p -- "$output_root"
output_root="$(cd -- "$output_root" && pwd -P)"
log_root="$output_root/slurm"
mkdir -p -- "$log_root"
manifest="$log_root/submissions.tsv"
if [[ -e "$manifest" ]]; then
    printf 'Output root already has a submission manifest; choose a fresh directory: %s\n' "$output_root" >&2
    exit 2
fi
studies=(conditional accuracy threshold dense_benchmarks isotropic_benchmarks)
for study in "${studies[@]}"; do
    if [[ -e "$output_root/$study/raw.csv.gz" || -e "$output_root/$study/metadata.json" ]]; then
        printf 'Refusing to overwrite existing study output: %s/%s\n' "$output_root" "$study" >&2
        exit 2
    fi
done

sbatch_options=(--parsable --export=ALL --nodes=1 --ntasks=1
                "--cpus-per-task=$cpus" "--mem=$memory" "--time=$walltime"
                "--chdir=$repo_root" "--output=$log_root/%x-%j.out"
                "--error=$log_root/%x-%j.err")
if [[ -n "$partition" ]]; then
    sbatch_options+=("--partition=$partition")
fi
if [[ -n "$account" ]]; then
    sbatch_options+=("--account=$account")
fi
if [[ -n "${extra_sbatch[*]:-}" ]]; then
    sbatch_options+=("${extra_sbatch[@]}")
fi
printf 'study\tjob_id\tsbatch_receipt\tsubmitted_utc\n' > "$manifest"
for study in "${studies[@]}"; do
    receipt="$(sbatch "${sbatch_options[@]}" \
        "--job-name=rs-$study" "$script_dir/run_study.sbatch" \
        "$study" "$repo_root" "$output_root" "$python_bin")"
    job_id="${receipt%%;*}"
    if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
        printf 'Unexpected sbatch receipt for %s: %s\n' "$study" "$receipt" >&2
        exit 1
    fi
    printf '%s\t%s\t%s\t%s\n' "$study" "$job_id" "$receipt" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$manifest"
    printf 'Submitted %-22s job %s\n' "$study" "$job_id"
done
printf 'Submission manifest: %s\n' "$manifest"
printf 'Results and logs: %s\n' "$output_root"
