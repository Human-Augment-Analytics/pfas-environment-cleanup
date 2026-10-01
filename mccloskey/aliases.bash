pf_print_line() {
    awk -v n="$1" 'NR == n'
}

pf_print_column_csv() {
    awk -F',' -v col="$1" '{print $col}'
}

pf_extract_cluster() {
    local line_no=$(( $1 + 2 ))

    export PF_LINE="$(
        pf_print_line "$line_no" < shivani_ml_models/cluster_centers.csv
    )"
    echo "PF_LINE: $PF_LINE"

    export PF_CLUSTER_NO="$(
        printf '%s\n' "$PF_LINE" | pf_print_column_csv 1
    )"
    echo "PF_CLUSTER_NO: $PF_CLUSTER_NO"

    export PF_CLUSTER_SMILE="$(
        printf '%s\n' "$PF_LINE" | pf_print_column_csv 12
    )"
    echo "PF_CLUSTER_SMILE: '$PF_CLUSTER_SMILE'"
}

pf_dft_wrapper_cluster_tfa() {
    [[ -z "${PF_CLUSTER_NO:-}" || -z "${PF_CLUSTER_SMILE:-}" ]] && { echo "Cluster vars not set"; return 1; }

    echo "About to run: python scripts/dft_wrapper.py --user jmccloskey30 --pfas-name tfa --pfas-smiles 'FC(F)(F)C(=O)O' --adsorbent-source smiles --submit-if-missing --cluster-root /home/hice1/jmccloskey30/test_runs --workflow-script /home/hice1/jmccloskey30/pfas-environment-cleanup/scripts/run_dft_workflow.sh --adsorbent-name 'c$PF_CLUSTER_NO' --case-name 'jmccloskey30-c$PF_CLUSTER_NO' --adsorbent-smiles '$PF_CLUSTER_SMILE'"

    read -rp "Run? [y/N] " x
    [[ "$x" =~ ^[Yy]$ ]] && python scripts/dft_wrapper.py --user jmccloskey30 --pfas-name tfa --pfas-smiles 'FC(F)(F)C(=O)O' --adsorbent-source smiles --submit-if-missing --cluster-root /home/hice1/jmccloskey30/test_runs --workflow-script /home/hice1/jmccloskey30/pfas-environment-cleanup/scripts/run_dft_workflow.sh --adsorbent-name "c$PF_CLUSTER_NO" --case-name "jmccloskey30-c$PF_CLUSTER_NO" --adsorbent-smiles "$PF_CLUSTER_SMILE"
}

pf_slurm_update_email() {
    scontrol update JobId="${1:?Usage: pf_update_email JOBID}" \
        MailUser=jmccloskey30@gatech.edu MailType=END,FAIL
}

pf_shortest_smiles() {
    mlr --csv put '$len = strlen($medoid_SMILES)' then sort -n len shivani_ml_models/cluster_centers.csv
}

alias pf_slurm_squeue="squeue -u $USER"
alias pf_slurm_running_ids='pf_slurm_squeue -t RUNNING -h -o "%A"'
pf_slurm_sstat_all() {
    local jobs
    jobs="$(pf_slurm_running_ids | sed 's/$/.batch/' | paste -sd, -)"
    [[ -n "$jobs" ]] && sstat -j "$jobs" "$@"
}
pf_slurm_sstat() {
    pf_slurm_sstat_all --parsable2 --format=JobID,AveCPU,MaxRSS,AveRSS,MaxDiskRead,MaxDiskWrite |
    awk -F'|' 'BEGIN{OFS="|"} NR==1{$3="MaxRSS_GiB"; $4="AveRSS_GiB"} NR>1{gsub(/K/,"",$3); gsub(/K/,"",$4); $3=sprintf("%.2f",$3/1048576); $4=sprintf("%.2f",$4/1048576)} 1' |
    column -t -s'|'
}
alias pf_slurm_recent_ids="ls -1r ~/outputs | sed -n 's/^slurm-\([0-9]\+\)\..*/\1/p' | uniq | head"
pf_slurm_recent_logs() {
    local id
    id="$(pf_slurm_recent_ids | head -n 1)"
    code ~/outputs/slurm-"$id".out ~/outputs/slurm-"$id".err
}

alias pf_slurm_srun_bash() {
    srun --jobid=${1:?Usage: pf_update_email JOBID} --overlap --pty bash
}