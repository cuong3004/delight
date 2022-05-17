mkdir -p checkpoints/mtet_en_vi
fairseq-train \
    data-bin/mtet_en_vi \
    --arch delight_transformer \
    --dropout 0.2 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer nag --clip-norm 0.1 \
    --lr 0.5 --lr-scheduler fixed --force-anneal 50 \
    --max-tokens 4000 \
    --save-dir checkpoints/mtet_en_vi
