

# python scripts/dft_wrapper.py --user jmccloskey30 --case-name jmccloskey30-test-20260911 --adsorbent-name ctab --pfas-name tfa --adsorbent-source smiles --submit-if-missing --cluster-root /home/hice1/jmccloskey30/test_runs --workflow-script /home/hice1/jmccloskey30/pfas-environment-cleanup/scripts/run_dft_workflow.sh --adsorbent-smiles 'CCCCCCCCCCCCCCCC[N+](C)(C)C.[Br-]' --pfas-smiles 'FC(F)(F)C(=O)O'

pf_update_email() {
    scontrol update JobId="${1:?Usage: update_email JOBID}" \
        MailUser=jmccloskey30@gatech.edu MailType=END,FAIL
}