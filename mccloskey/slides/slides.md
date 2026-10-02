---
theme: default
title: Making the DFT Workflow Easier to Run
author: John McCloskey
info: |
  Eight demonstrated helpers and one comparison with existing automation.
  Branch: mccloskey/run_simulations_20261001 through c31f829.
layout: default
aspectRatio: 16/9
colorSchema: light
routerMode: hash
fonts:
  sans: Arial
  mono: monospace
  provider: none
defaults:
  layout: default
---

<div class="eyebrow">1 / 9 · Select · 12 seconds</div>

# Find short candidate SMILES

<div class="columns">
<div>
<div class="panel-label">Function · line wrapped</div>

```bash
pf_shortest_smiles() {
  mlr --csv put \
    '$len = strlen($medoid_SMILES)' \
    then sort -n len \
    shivani_ml_models/cluster_centers.csv
}
```

<div class="annotation">Add a length column, then sort smallest first.</div>
</div>
<div>
<div class="panel-label">Output · selected columns and first four rows</div>

```text
$ pf_shortest_smiles | head
cluster  n_points  medoid_SMILES  len
553      57        C(NC(=S)N)O    11
809      65        CC1C(O1)C=C    11
58       8         C=CC(=O)[O-]   12
147      74        C1CC1N=C(N)N   12
```

<div class="annotation">Cluster 553 is a short starting candidate.</div>
</div>
</div>

<div class="callout">John's focus: make the existing workflow easier to run and inspect, starting with a simple candidate.</div>
<div class="source">mccloskey/aliases.bash · Supplied local output from john@brainiac</div>

<!--
12 seconds: "John is improving his own ergonomics for running the existing
workflow. This helper puts shorter candidate structures first, so it is easy
to pick a starting case. Here that is cluster 553."
SMILES length is a selection heuristic, not a runtime prediction.
Selected columns and rows retain the supplied values.
-->

---

<div class="eyebrow">2 / 9 · Select · 12 seconds</div>

# Load cluster 553 into shell variables

<div class="columns">
<div>
<div class="panel-label">Function excerpt · key steps</div>

```bash
pf_extract_cluster() {
  local line_no=$(( $1 + 2 ))
  # ... read that row into PF_LINE ...
  export PF_CLUSTER_NO="$(
    printf '%s\n' "$PF_LINE" |
      pf_print_column_csv 1)"
  export PF_CLUSTER_SMILE="$(
    printf '%s\n' "$PF_LINE" |
      pf_print_column_csv 12)"
}
```

</div>
<div>
<div class="panel-label">Output excerpt · long PF_LINE omitted</div>

```text
$ pf_extract_cluster 553
PF_CLUSTER_NO: 553
PF_CLUSTER_SMILE: 'C(NC(=S)N)O'
```

<div class="annotation">Column 1 → cluster ID.<br>Column 12 → medoid SMILES.</div>
<div class="annotation">+2 accounts for the header and one-based line numbers.</div>
</div>
</div>

<div class="source">Input is a data-row index; index 553 currently matches cluster ID 553 · Supplied local output</div>

<!--
12 seconds: "The next helper remembers the selected candidate's ID and
structure. John can reuse them in the next command instead of copying a long
CSV row by hand."
pf_print_line uses awk NR == n; pf_print_column_csv uses awk -F','. This is a
row-index lookup, not a search for a cluster ID, and assumes the CSV layout.
-->

---

<div class="eyebrow">3 / 9 · Submit · 15 seconds</div>

# Preview and confirm the TFA run

<div class="columns">
<div>
<div class="panel-label">Function excerpt · fixed arguments omitted</div>

```bash
pf_dft_wrapper_cluster_tfa() {
  # ... check variables; print preview ...
  read -rp "Run? [y/N] " x
  [[ "$x" =~ ^[Yy]$ ]] && \
    python scripts/dft_wrapper.py \
    --adsorbent-name "c$PF_CLUSTER_NO" \
    --case-name "jmccloskey30-c$PF_CLUSTER_NO" \
    --adsorbent-smiles "$PF_CLUSTER_SMILE"
  # Fixed arguments omitted above:
  # user, TFA, source, and cluster paths.
}
```

<div class="annotation">Selected variables become the case name and molecular input.</div>
</div>
<div>
<div class="panel-label">Output excerpt · preview line wrapped</div>

```text
$ pf_dft_wrapper_cluster_tfa
About to run: python scripts/dft_wrapper.py
  ...
  --pfas-name tfa
  --pfas-smiles 'FC(F)(F)C(=O)O'
  --adsorbent-name 'c553'
  --case-name 'jmccloskey30-c553'
  --adsorbent-smiles 'C(NC(=S)N)O'
Run? [y/N]
```

<div class="annotation">Review the command; only y or Y runs it.</div>
</div>
</div>

<div class="source">mccloskey/aliases.bash · Supplied output ends at the confirmation prompt</div>

<!--
15 seconds: "This fills in the candidate and TFA inputs, then shows the command
before running it. It saves typing and gives John a quick check before saying
yes. The following slides show his running jobs."
This explanatory excerpt omits fixed arguments and is not a replacement for
the real function. The full version passes --submit-if-missing and fixed
user/source/root/workflow/PFAS options. This capture does not show a response
or job receipt, so it alone does not prove submission.
-->

---

<div class="eyebrow">4 / 9 · Monitor · 8 seconds</div>

# See the user's queued jobs

<div class="panel-label">Alias</div>

```bash
alias pf_slurm_squeue="squeue -u $USER"
```

<div class="panel-label">Supplied cluster output · spacing condensed</div>

```text
$ pf_slurm_squeue
  JOBID PARTITION     NAME     USER ST    TIME NODES NODELIST(REASON)
6029069   coc-cpu dft_jmcc jmcclosk  R 8:27:33     1 atl1-1-02-004-15-2
6029068   coc-cpu dft_jmcc jmcclosk  R 8:27:47     1 atl1-1-02-003-19-1
```

<div class="callout">R means running. Both jobs have one node and about 8½ hours of elapsed time.</div>
<div class="source">mccloskey/aliases.bash · Supplied login-ice-gnr-1 output</div>

<!--
8 seconds: "One short command shows John's jobs. Both are running on one
node each, for about eight and a half hours."
Historical user-supplied snapshot, not a fresh cluster query.
-->

---

<div class="eyebrow">5 / 9 · Monitor · 8 seconds</div>

# Get job IDs for the next command

<div class="panel-label">Alias</div>

```bash
alias pf_slurm_running_ids='pf_slurm_squeue -t RUNNING -h -o "%A"'
```

<div class="columns">
<div>
<div class="annotation">RUNNING filters jobs.<br>-h hides the header.<br>%A prints job IDs.</div>
</div>
<div>
<div class="panel-label">Supplied cluster output</div>

```text
$ pf_slurm_running_ids
6029069
6029068
```

</div>
</div>

<div class="callout">Small helpers compose: queue → running IDs → resource statistics.</div>
<div class="source">mccloskey/aliases.bash · Supplied login-ice-gnr-1 output</div>

<!--
8 seconds: "This returns just the running job numbers. Other helpers can use
that list automatically, saving another copy-and-paste step."
-->

---

<div class="eyebrow">6 / 9 · Monitor · 17 seconds</div>

# Read memory and I/O in GiB

<div class="panel-label">Function excerpt · awk formatting omitted</div>

```bash
pf_slurm_sstat() {
  pf_slurm_sstat_all --parsable2 \
    --format=JobID,AveCPU,MaxRSS,AveRSS,MaxDiskRead,MaxDiskWrite |
    # ... awk converts RSS and disk counters to GiB ...
    column -t -s'|'
}
```

<div class="panel-label">Supplied cluster output</div>

```text
$ pf_slurm_sstat
JobID          AveCPU    MaxRSS_GiB  AveRSS_GiB  DiskRead_GiB  DiskWrite_GiB
6029069.batch  08:25:18  11.40       2.95        0.13          3.50
6029068.batch  08:25:50  23.31       6.23        0.13          1.91
```

<div class="callout">Job 6029068 reports about twice the maximum RSS: 23.31 GiB versus 11.40 GiB.</div>
<div class="source">mccloskey/aliases.bash · Updated user-supplied snapshot · October 1, 2026</div>

<!--
17 seconds: "This makes memory and disk activity easier to read. The second
job reports about twice the maximum memory use. John can quickly see which
run deserves closer inspection."
pf_slurm_sstat_all appends .batch to IDs, joins with commas, and calls sstat.
The omitted awk strips K from RSS and divides by 1048576, divides disk counters
by 1073741824, relabels headers, and formats two decimal places. This excerpt
is not runnable. AveCPU is CPU time, not elapsed time; MaxRSS is not allocation
memory. Disk columns are maximum task counters, not summed job I/O.
-->

---

<div class="eyebrow">7 / 9 · Diagnose · 17 seconds</div>

# Inspect the calculation inside its job

<div class="panel-label">Function excerpt · outer loop and PID parsing omitted</div>

```bash
pf_slurm_processes() {
  # ... find running jobs; srun once per node ...
  listing=$(scontrol listpids "$1")
  # ... extract job PIDs into "$pids" ...
  ps -ww -p "$pids" \
    -o pid,ppid,stat,etime,time,pcpu,pmem,rss,args --sort=-rss
}
```

<div class="panel-label">Output excerpt · selected fields from the pw.x row</div>

```text
$ pf_slurm_processes
Job 6029069
0: Node: atl1-1-02-004-15-2.pace.gatech.edu
0: ELAPSED   TIME     %CPU  RSS      COMMAND
0: 08:28:06  08:25:48 99.5  2207500  pw.x -in adsorbent.in
```

<div class="callout">c553 is still calculating the adsorbent. Check QE output for convergence.</div>
<div class="source">October 1, 2026 · Prolog 21:29:49 · Parent command identifies c553 · ps RSS is in KiB</div>

<!--
17 seconds: "This shows the actual calculation inside the job. The adsorbent
calculation is still using about one CPU. It is a useful activity check;
John still needs the calculation output to confirm a finished result."
The excerpt shows the original inner commands without their outer context.
Original srun uses --overlap --exact --immediate=10 with one task/CPU per node
and --label. Output retains exact supplied values for selected fields, omitting
PID, PPID, STAT, %MEM, other processes, and Slurm prolog lines. Parent command
shows jmccloskey30-c553, mpirun -np 1, and --skip-pfas. Not runnable as shown.
-->

---

<div class="eyebrow">8 / 9 · Diagnose · 13 seconds</div>

# Check the size of output files

<div class="columns">
<div>
<div class="panel-label">Function</div>

```bash
pf_largest_files() {
  find . -type f -printf '%s %p\n' |
    sort -nr |
    head -n 30 |
    numfmt --field=1 --to=iec
}
```

<div class="annotation">Sort files by bytes; show the largest 30 with readable sizes.</div>
</div>
<div>
<div class="panel-label">Output excerpt · paths shortened</div>

```text
# From test_runs
$ pf_largest_files
177M .../c100/.../charge-density.hdf5
88M  .../c553/.../charge-density.hdf5
72M  .../tfa/.../charge-density.hdf5
34M  .../tfa/.../wfc1.hdf5
```

<div class="annotation">Charge densities and wavefunctions are the largest files shown.</div>
</div>
</div>

<div class="callout">Output-file sizes matter: batch runs multiply these artifacts and can run out of storage space.</div>
<div class="source">Supplied test_runs output · Committed sample log also reports “Disk quota exceeded”</div>

<!--
13 seconds: "The biggest files here are calculation artifacts.
Checking output sizes matters: many batch cases multiply these artifacts and
can run out of space. We already have a disk-quota failure in a sample log."
Full paths are under compounds/adsorbents/c100/Outputs/adsorbent.save,
compounds/adsorbents/c553/Outputs/adsorbent.save, and
compounds/pfas/tfa/Outputs/pfas.save. First four supplied entries shown.
Source of quota failure: scripts/sample_logs/simple_screen_5898218_9.out.
Largest files alone are not a quota or total-storage measurement.
Do not infer that c100 belongs to job 6029068 from these separate logs.
-->

---

<div class="eyebrow">9 / 9 · Reuse existing tooling · 18 seconds</div>

# Some of this workflow already exists

<div class="columns">
<div>
<div class="panel-label">run_batch_screening.sh · excerpt</div>

```bash
#SBATCH --array=2-26
# ... parse one CSV row per task ...
export CASE_NAME="${ADS_NAME}_TFA"
export ADSORBENT_NAME="$ADS_NAME"
export ADSORBENT_SMILES="$ADS_SMILES"
export PFAS_NAME="TFA"
export PFAS_SMILES="FC(F)(F)C(=O)O"
bash run_dft_workflow.sh
```

<div class="annotation">Already selects candidates, names cases, and launches TFA runs in a batch.</div>
</div>
<div>
<div class="panel-label">run_dft_workflow.sh · excerpt</div>

```bash
MPI_TASKS="${SLURM_NTASKS:-1}"
# ... configure MPI and build ARGS ...
conda run -p "$ENV_PREFIX" python \
  qespresso_pipeline/run_adsorption_case.py \
  "${ARGS[@]}"
```

<div class="annotation">Already prepares the environment, configures MPI, and runs the shared pipeline.</div>
</div>
</div>

<div class="callout">Some setup overlaps with existing scripts. John's contribution here is convenience for his own day-to-day workflow.</div>
<div class="source">Sources: scripts/run_batch_screening.sh · scripts/run_dft_workflow.sh · Excerpts omit setup and validation</div>

<!--
18 seconds: "Some of this repeats existing tooling. Batch screening already
selects candidates, and the workflow script runs the calculations. John's
focus is his own ergonomics: less typing, easier checks, and better visibility
into jobs and storage. These helpers sit around the existing workflow."
Batch input is simple_adsorbents.csv, not the medoid CSV, so candidate libraries
differ. The array has 25 tasks, each requesting 4 MPI tasks and 192G memory;
that is RAM, not a disk-storage allowance. Actual directives take precedence
over stale comments mentioning 100 rows and 64G. It uses Python csv.reader,
which is more robust than the aliases' awk comma splitting. The aliases also
call the existing dft_wrapper.py and shared workflow rather than implementing
the DFT pipeline again. No claim that these scripts already inspect file sizes.
Suggested timing totals 120 seconds across nine slides.
-->
