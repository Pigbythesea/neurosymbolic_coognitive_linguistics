#!/bin/bash
#SBATCH --job-name=neurosym-data
#SBATCH --account=tshu2_scai01
#SBATCH --qos=jhu
#SBATCH --partition=med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --comment=accept_cost
#SBATCH --dependency=singleton
#SBATCH --chdir=/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics
#SBATCH --output=/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics/logs/download-%j.log

set -euo pipefail
trap 'result=$?; printf "JOB EXIT status=%s job=%s\n" "$result" "${SLURM_JOB_ID:-unset}"' EXIT

source /weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics/scripts/cluster_env.sh
cd "$NEUROSYM_ROOT"
printf 'Job %s on %s; source manifest SHA-256:\n' "$SLURM_JOB_ID" "$(hostname -s)"
sha256sum manifests/deniz-reading.json
/usr/bin/python3.11 --version

# No packages or environment installation are needed for acquisition.
/usr/bin/python3.11 -I -B -u scripts/acquire_data.py download \
    --manifest manifests/deniz-reading.json --profile reading-core \
    --jobs 4 --data-root "$NEUROSYM_ROOT/data"
/usr/bin/python3.11 -I -B -u scripts/acquire_data.py verify \
    --manifest manifests/deniz-reading.json --profile reading-core --data-root "$NEUROSYM_ROOT/data"

printf '%s\n' 'CLUSTER DOWNLOAD AND VERIFICATION COMPLETE'
