# Fine-tuning BART on COVID Dialogue Dataset #

## 1) Introduction

BART model [https://arxiv.org/pdf/1910.13461.pdf](https://arxiv.org/pdf/1910.13461.pdf)

Fairseq [https://github.com/pytorch/fairseq](https://github.com/pytorch/fairseq)

Fairseq tutorial on fine-tuning BART on Seq2Seq task [https://github.com/pytorch/fairseq/blob/master/examples/bart/README.summarization.md](https://github.com/pytorch/fairseq/blob/master/examples/bart/README.summarization.md)

COVID Dialogue Dataset [https://github.com/UCSD-AI4H/COVID-Dialogue](https://github.com/UCSD-AI4H/COVID-Dialogue)

## 2) Download model

Download the BART-large model from [here](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz "here")

Data is already in this Repo

put the model at

    REPO ROOT
	 |
	 |-- bart.large
	 |	  |-- dict.txt
	 |	  |-- model.pt
	 |	  |-- NOTE
	 |-- data
	 |	  |...
	 |-- preprocess_data
	 |	  |...
	 |...

## 3) Fine-tuning
Prerequisite:

**PyTorch**

**Fairseq** (to install, follow the guidance in [here](https://github.com/pytorch/fairseq). In most cases, just simply run "pip install fairseq")

During fine-tuning, the input is what the patients said and output is what the doctors said. Thus the model is playing a role of a doctor.

Data is already preprocessed. if you would like to preprocess again, you can run the file in preprocess_data directory in this order:

    python preprocess_data.py
	bash bpe.sh
	bash binarize.sh

Then fine-tuning the using train.sh in repo root directory. Before using it, edit this file to fit into your own machine. With the default setting, the model is fine-tuning on 6 GPUs and consuming around 10G GPU memory of each (totally 60G GPU memory). You can change MAX_TOKENS flag to adjust batch size. (fine more information about command-line tools at [here](https://fairseq.readthedocs.io/en/latest/command_line_tools.html)

After adjustment, you simply run this command:

    bash train.sh

A checkpoint will be dumped at ./checkpoints/checkpoint_last.pt every epoch. You can stop fine-tuning whenever you want. Note that from my empirical experiments, too many epochs may lead to bad performance when doing inference.

## 4) Interact with your model

run the command:

    bash interactive.sh

Example output:

    Hi, doctor, what are the symptoms of covid-19?
	S-2     17250 11 6253 11 644 389 262 7460 286 39849 312 12 1129 30
	H-2     -0.13718903064727783    Symptoms. The symptom of COVID-19 begins with mild flu-like symptoms such as fatigue, sore throat and sneeze, followed by fever, dry cough. In severe cases, the cough can progress to productive cough, persistent and followed by shortness of breath. Some patients may also experience GI symptoms such as nausea vomiting and diarrhea.
	P-2     -0.9147 -0.0561 -0.1125 -1.7157 -1.1783 -0.1116 -0.0821 -0.0716 -0.0983 -0.0878 -0.2107 -0.0986 -0.0958 -0.0830 -0.0978 -0.0701 -0.0765 -0.1056 -0.1026 -0.0833 -0.1028 -0.0383 -0.0865 -0.1034 -0.0318 -0.0772 -0.0434 -0.1055 -0.0736 -0.0976 -0.0637 -0.1042 -0.0476 -0.0865 -0.1019 -0.0898 -0.0652 -0.0786 -0.1048 -0.1022 -0.0760 -0.0950 -0.0551 -0.0991 -0.0442 -0.0816 -0.1027 -0.0517 -0.1037 -0.0623 -0.0988 -0.0622 -0.0796 -0.1010 -0.0761 -0.1033 -0.1186 -0.0735 -0.1044 -0.0949 -0.1168 -0.0729 -0.0807 -0.1026 -0.1003 -0.0387 -0.0440 -0.1044 -0.0445 -0.1034 -0.1913

If you think extra output is annoying, you can write a interact script by yourself, following the guidance below.

## 5) Use the model in your code

This is from [fairseq tutorial](https://github.com/pytorch/fairseq/blob/master/examples/bart/README.summarization.md), from which you can learn how to use the model.

    import torch
	from fairseq.models.bart import BARTModel

	bart = BARTModel.from_pretrained(
    'checkpoints/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='cnn_dm-bin'
	)

	bart.cuda()
	bart.eval()
	bart.half()
	count = 1
	bsz = 32
	with open('cnn_dm/test.source') as source, open('cnn_dm/test.hypo', 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    for sline in source:
        if count % bsz == 0:
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)

            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
            slines = []

        slines.append(sline.strip())
        count += 1
    if slines != []:
        hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis + '\n')
            fout.flush()

Find more information from [fairseq bart repo](https://github.com/pytorch/fairseq/tree/master/examples/bart)!