for LANG in source target
do
  python -m multiprocessing_bpe_encoder \
  --encoder-json encoder.json \
  --vocab-bpe vocab.bpe \
  --inputs "patient2doctor/train.$LANG" \
  --outputs "patient2doctor/train.bpe.$LANG" \
  --workers 60 \
  --keep-empty;
done