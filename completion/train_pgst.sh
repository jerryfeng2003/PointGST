bash ./scripts/train.sh 0 \
    --config cfgs/PCN_models/AdaPoinTr_pgst.yaml \
    --exp_name pgst \
    --start_ckpts ckpts/AdaPoinTr_ps55.pth \
    --model pgst