# Example site: Iridis X (University of Southampton)

A worked example of running CellWarden on a Slurm cluster with:
- no FUSE (so `.sif` files can't be mounted directly);
- no internet on the compute nodes;
- file-count quotas.

The same steps apply to many HPC systems.

## 1. Put the image in a shared folder, once, on the login node

```bash
module load apptainer/1.4.0
export APPTAINER_TMPDIR=/tmp/$USER-apptainer APPTAINER_CACHEDIR=/path/to/project/apptainer-cache
apptainer build --sandbox /tmp/$USER-sbx docker://ghcr.io/cellwarden/cellwarden:0.3.0
rsync -a /tmp/$USER-sbx/ /path/to/project/images/cellwarden_sandbox/ && rm -rf /tmp/$USER-sbx
```

- **Why a sandbox folder:** without FUSE a `.sif` needs `--unsquash`, which unpacks it into the
  node's RAM-backed `/tmp` on every start. Jobs then need ~96 GB of memory and 30 s more. An
  unpacked sandbox (~133k files, 12 GB) runs directly.
- **Why `rsync -a`:** unpacking straight onto GPFS loses the execute permissions, because new
  files inherit the folder's ACL. `rsync -a` sets each file's mode.
- **Where to put it:** in a project folder, not home or scratch. Those have file-count quotas
  (Iridis: home 204,800 files, scratch 614,400).

After a code change (developers), update the sandbox in place instead of rebuilding:
`containers/update_sandbox.sh /path/to/project/images/cellwarden_sandbox --code-only`.

## 2. A short command

```bash
cellwarden() { apptainer exec --cleanenv --env-file "${CELLWARDEN_ENV:-/dev/null}" -B /iridisfs,/scratch /path/to/project/images/cellwarden_sandbox cellwarden "$@"; }
```

`CELLWARDEN_ENV=.env cellwarden doctor` checks the install and your LLM key (GETTING_STARTED.md section 3).

## 3. Small runs: the login node

PBMC 3k takes about 6 minutes end to end, so small studies can run on the login node.
CellWarden runs its stages under `nice` and caps threads (`resources.threads`). Use `--detach`
for long runs, so they survive a dropped SSH session.

## 4. Big runs: login node + compute node

The review agents and Open Targets need internet, which Iridis compute nodes don't have. So
split the run into three steps that share the run folder:

```bash
# 1. login node: questions, ingest, QC and the QC review
cellwarden run data/*.h5 -i --until qc-review --out /scratch/$USER/my_study

# 2. compute node: normalisation, batch correction, clustering, UMAP, markers, CellTypist (offline)
export CELLWARDEN_IMAGE=/path/to/project/images/cellwarden_sandbox CELLWARDEN_BIND=/iridisfs,/scratch
sbatch --export=ALL,RUN=/scratch/$USER/my_study conf/jobs/slurm_compute_offline.sbatch

# 3. login node, after the job: context and annotation reviews, DE + Open Targets, interpretation, report
cellwarden resume --run-dir /scratch/$USER/my_study
```

- **Job resources:** edit the `#SBATCH` lines of `conf/jobs/slurm_compute_offline.sbatch`. The
  defaults are 8 cores, 64 GB and 4 hours on `amd`.
- **GPU:** use `--partition=l4 --gres=gpu:1` and add `--nv` after `apptainer exec`.

## 5. Site file

[`conf/sites/iridisx.yaml`](../../conf/sites/iridisx.yaml) holds the Iridis defaults: resource
caps, the GPU memory cap for the shared L4, and the scheduler settings used by
`execution: slurm | split`. Point a config at it with `site: iridisx`.
