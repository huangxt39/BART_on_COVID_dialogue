fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "patient2doctor/train.bpe" \
  --destdir "patient2doctor-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;