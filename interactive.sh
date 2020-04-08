MODEL_DIR=checkpoints/
fairseq-interactive \
    --path $MODEL_DIR/checkpoint_last.pt preprocess_data/patient2doctor-bin \
    --beam 5 --source-lang source --target-lang target \
    --tokenizer space \
    --bpe gpt2 --gpt2-encoder-json preprocess_data/encoder.json --gpt2-vocab-bpe preprocess_data/vocab.bpe