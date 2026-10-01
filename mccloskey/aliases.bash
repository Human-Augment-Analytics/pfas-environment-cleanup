

alias pf_dft_wrapper_tfa="python scripts/dft_wrapper.py --user jmccloskey30  --pfas-name tfa --pfas-smiles 'FC(F)(F)C(=O)O' --adsorbent-source smiles --submit-if-missing --cluster-root /home/hice1/jmccloskey30/test_runs --workflow-script /home/hice1/jmccloskey30/pfas-environment-cleanup/scripts/run_dft_workflow.sh" 

alias pf_update_email="scontrol update JobId="${1:?Usage: update_email JOBID}" \
        MailUser=jmccloskey30@gatech.edu MailType=END,FAIL
}
