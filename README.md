# PremiseGeneratorBert

Use `enviorment.yml` file with conda to install neccecery modules:

```
 conda env update -f environment.yml
```

Then, clone and install this fork of `Huggingface` according to the instruction in this git: <https://github.com/dimi1357/transformers.git>.

Availble commands (python main.py -h to see this menu)

```
usage: main.py [-h] {train,test,generate,pipeline} ...

Experiments

positional arguments:
  {train,test,generate,pipeline}
                        Sub-commands
    train               Train a model
    test                Test model on test set
    generate            Generate new dataset
    pipeline            Pipeline
```

Quick examples:

Train generative BART model on SNLI:

```
python -n NAME --checkpoints <path to checkpoints> --model type bart --model-name facebook/bart-base --data-dir-prefix data/snli_1.0/cl_snli
```

Train discriminative BART model on MNLI:

```
python -n NAME --checkpoints <path to checkpoints> --model type disc --model-name facebook/bart-base --data-dir-prefix data/mnli/cl_multinli
```

Train hypothesis-only BERT model on MNLI:

```
python -n NAME --checkpoints <path to checkpoints> --model type disc -ho --model-name bert-base-uncased --data-dir-prefix data/mnli/cl_multinli
```

Full list of arguments:

**Train**:

```
usage: main.py train [-h] --run-name RUN_NAME [--out-dir OUT_DIR] [--seed SEED] [--drive DRIVE]
                     [--do-test DO_TEST] [--gen-premise GEN_PREMISE] [--inject-bias INJECT_BIAS]
                     [--bias-ids BIAS_IDS [BIAS_IDS ...]] [--bias-ratio BIAS_RATIO]
                     [--bias-location BIAS_LOCATION] [--non-discriminative-bias]
                     [--attribution-map ATTRIBUTION_MAP] [--filt-method FILT_METHOD]
                     [--move-to-hypothesis] [--bs-train BATCH_SIZE] [--bs-test BATCH_SIZE]
                     [--batches BATCHES] [--epochs EPOCHS] [--early-stopping EARLY_STOPPING]
                     [--checkpoints CHECKPOINTS] [--lr LR] [--reg REG]
                     [--data-dir-prefix DATA_DIR_PREFIX] [--max-len MAX_LEN]
                     [--decoder-max-len DECODER_MAX_LEN] [--param-freezing-ratio PARAM_FREEZING_RATIO]
                     [--optimizer-type OPTIMIZER_TYPE] [--sched SCHED] [--reduction REDUCTION]
                     [--momentum MOMENTUM] [--word-dropout WORD_DROPOUT]
                     [--label-smoothing-epsilon LABEL_SMOOTHING_EPSILON]
                     [--hyp-only-model HYP_ONLY_MODEL] [--attribution-tokenizer ATTRIBUTION_TOKENIZER]
                     [--threshold THRESHOLD] [--train-hyp] [--test-with-prior] [--cheat] [--calc-uniform]
                     [--tie-embeddings] [--hypothesis-only] [--premise-only] [--gradual-unfreeze]
                     [--generate-hypothesis] [--hard-validation] [--merge-train] [--label LABEL]
                     [--reverse] [--tie-encoder-decoder] [--pure-gen] [--model-path MODEL_PATH]
                     [--model-name MODEL_NAME] [--model-type MODEL_TYPE]
                     [--decoder-model-name DECODER_MODEL_NAME] [--beta1 BETA1] [--beta2 BETA2]
                     [--epsilon EPSILON] [--weight-decay WEIGHT_DECAY] [--gamma GAMMA]

optional arguments:
  -h, --help            show this help message and exit
  --run-name RUN_NAME, -n RUN_NAME
                        Name of run and output file
  --out-dir OUT_DIR, -o OUT_DIR
                        Output folder
  --seed SEED, -s SEED  Random seed
  --drive DRIVE, -d DRIVE
                        Pass "True" if you are running this on Google Colab
  --do-test DO_TEST, -t DO_TEST
                        Pass "True" if you want to run a test on test set
  --gen-premise GEN_PREMISE
  --inject-bias INJECT_BIAS
                        Select number of labels to inject bias to their corresponding hypotheses
  --bias-ids BIAS_IDS [BIAS_IDS ...]
                        Select the ids of the biases symbols
  --bias-ratio BIAS_RATIO
                        Select the percentege of labels to inject bias to their corresponding hypotheses
  --bias-location BIAS_LOCATION
                        Select where in the hypotheses to inject the bias, can be either "start" or
                        "end", otherwise will be random location
  --non-discriminative-bias, -ndb
                        Make the synthetic bias non-discriminative
  --attribution-map ATTRIBUTION_MAP, -am ATTRIBUTION_MAP
                        path of attribution maps folder
  --filt-method FILT_METHOD, -fm FILT_METHOD
                        The method to filter the premis by. Should be in [sum,mean,max,max-abs,min-
                        abs,true,rand
  --move-to-hypothesis, -mth
                        Move the filtered words from the premise to the hypothesis
  --bs-train BATCH_SIZE
                        Train batch size
  --bs-test BATCH_SIZE  Test batch size
  --batches BATCHES     Number of batches per epoch
  --epochs EPOCHS       Maximal number of epochs
  --early-stopping EARLY_STOPPING
                        Stop after this many epochs without improvement
  --checkpoints CHECKPOINTS
                        Save model checkpoints to this file when test accuracy improves
  --lr LR, -lr LR       Learning rate
  --reg REG             L2 regularization
  --data-dir-prefix DATA_DIR_PREFIX
                        Prefix of the path to data
  --max-len MAX_LEN, -ml MAX_LEN
                        Length of longest sequence (or bigger), 0 if you don't know
  --decoder-max-len DECODER_MAX_LEN, -dml DECODER_MAX_LEN
                        Length of longest sequence of the decoder (or bigger), 0 if you don't know
  --param-freezing-ratio PARAM_FREEZING_RATIO
                        How many of the params to freeze
  --optimizer-type OPTIMIZER_TYPE, -ot OPTIMIZER_TYPE
                        Which type of optimizer to use
  --sched SCHED         Which type of optimizer to use
  --reduction REDUCTION, -reduce REDUCTION
                        How to reduce loss, can be "sum" or "mean"
  --momentum MOMENTUM, -m MOMENTUM
                        Momentum for SGD
  --word-dropout WORD_DROPOUT, -wdo WORD_DROPOUT
                        Word dropout rate during training
  --label-smoothing-epsilon LABEL_SMOOTHING_EPSILON, -lse LABEL_SMOOTHING_EPSILON
                        Epsilon argument for label smoothing (does not uses labels smoothing by default
  --hyp-only-model HYP_ONLY_MODEL, -hom HYP_ONLY_MODEL
                        If you want to weigh loss by htpothesis only output
  --attribution-tokenizer ATTRIBUTION_TOKENIZER, -at ATTRIBUTION_TOKENIZER
                        Huggingface model name for the attributions, default is same as encoder
  --threshold THRESHOLD, -th THRESHOLD
  --train-hyp
  --test-with-prior, -twp
  --cheat
  --calc-uniform, -cu
  --tie-embeddings, -te
  --hypothesis-only, -ho
  --premise-only, -po
  --gradual-unfreeze, -gu
  --generate-hypothesis, -gh
  --hard-validation, -hv
  --merge-train
  --label LABEL, -l LABEL
                        Create generative model only for one label
  --reverse, -rev       Generate hypothesis
  --tie-encoder-decoder, -ted
  --pure-gen, -pg
  --model-path MODEL_PATH
                        Path to fined-tuned model
  --model-name MODEL_NAME
                        Name of the huggingface model
  --model-type MODEL_TYPE
                        Type of the model (encode-decode or hybrid)
  --decoder-model-name DECODER_MODEL_NAME
                        Name of the decoder, if empty then same as encoder
  --beta1 BETA1, -b1 BETA1
  --beta2 BETA2, -b2 BETA2
  --epsilon EPSILON, -eps EPSILON
  --weight-decay WEIGHT_DECAY, -wd WEIGHT_DECAY
  --gamma GAMMA
```

**Test**:

```
usage: main.py test [-h] --run-name RUN_NAME [--out-dir OUT_DIR] [--seed SEED]
                    [--save-results SAVE_RESULTS] [--reduction REDUCTION] [--filt-method FILT_METHOD]
                    [--bs-test BATCH_SIZE] [--batches BATCHES] [--data-dir-prefix DATA_DIR_PREFIX]
                    [--max-len MAX_LEN] [--decoder-max-len DECODER_MAX_LEN] [--create-premises]
                    [--attribution-map ATTRIBUTION_MAP] [--threshold THRESHOLD]
                    [--attribution-tokenizer ATTRIBUTION_TOKENIZER] [--move-to-hypothesis]
                    [--hyp-only-model HYP_ONLY_MODEL] [--test-with-prior] [--calc-uniform] [--reverse]
                    [--inject-bias INJECT_BIAS] [--bias-ids BIAS_IDS [BIAS_IDS ...]]
                    [--bias-ratio BIAS_RATIO] [--bias-location BIAS_LOCATION] [--non-discriminative-bias]
                    [--model-name MODEL_NAME] [--model-path MODEL_PATH] [--model-type MODEL_TYPE]
                    [--checkpoints CHECKPOINTS] [--decoder-model-name DECODER_MODEL_NAME] [--label LABEL]
                    [--hypothesis-only] [--premise-only] [--pure-gen] [--generate-hypothesis]
                    [--save-likelihoods SAVE_LIKELIHOODS]

optional arguments:
  -h, --help            show this help message and exit
  --run-name RUN_NAME, -n RUN_NAME
                        Name of run and output file
  --out-dir OUT_DIR, -o OUT_DIR
                        Output folder
  --seed SEED, -s SEED  Random seed
  --save-results SAVE_RESULTS, -sr SAVE_RESULTS
                        Pass path if you want to save the results
  --reduction REDUCTION, -reduce REDUCTION
                        How to reduce loss, can be "sum" or "mean"
  --filt-method FILT_METHOD, -fm FILT_METHOD
  --bs-test BATCH_SIZE  Test batch size
  --batches BATCHES     Number of batches per epoch, pass "0" if you want the full database
  --data-dir-prefix DATA_DIR_PREFIX
                        Prefix of the path to data
  --max-len MAX_LEN     Length of longest sequence (or bigger), 0 if you don't know
  --decoder-max-len DECODER_MAX_LEN, -dml DECODER_MAX_LEN
                        Length of longest sequence of the decoder (or bigger), 0 if you don't know
  --create-premises, -cp
  --attribution-map ATTRIBUTION_MAP, -am ATTRIBUTION_MAP
                        path of attribution maps folder
  --threshold THRESHOLD, -th THRESHOLD
  --attribution-tokenizer ATTRIBUTION_TOKENIZER, -at ATTRIBUTION_TOKENIZER
                        Huggingface model name for the attributions, default is same as encoder
  --move-to-hypothesis, -mth
  --hyp-only-model HYP_ONLY_MODEL, -hom HYP_ONLY_MODEL
                        If you want to weigh loss by htpothesis only output
  --test-with-prior, -twp
  --calc-uniform, -cu
  --reverse, -rev       Generate hypothesis
  --inject-bias INJECT_BIAS
                        Select number of labels to inject bias to their corresponding hypotheses
  --bias-ids BIAS_IDS [BIAS_IDS ...]
                        Select the ids of the biases symbols
  --bias-ratio BIAS_RATIO
                        Select the percentege of labels to inject bias to their corresponding hypotheses
  --bias-location BIAS_LOCATION
                        Select where in the hypotheses to inject the bias, can be either "start" or
                        "end", otherwise will be random location
  --non-discriminative-bias, -ndb
                        Make the synthetic bias non-discriminative
  --model-name MODEL_NAME
                        Name of the huggingface model
  --model-path MODEL_PATH
                        Path to fined-tuned model
  --model-type MODEL_TYPE
                        Type of the model (encode-decode or hybrid)
  --checkpoints CHECKPOINTS
                        Checkpoint to torch model
  --decoder-model-name DECODER_MODEL_NAME
                        Only if encoder and decoder are different
  --label LABEL, -l LABEL
                        Create generative model only for one label
  --hypothesis-only, -ho
  --premise-only, -po
  --pure-gen, -pg
  --generate-hypothesis, -gh
  --save-likelihoods SAVE_LIKELIHOODS, -sl SAVE_LIKELIHOODS
                        Pass path if you want to save the likelihoods as a torch tensor
```

**Pipeline**:

```
usage: main.py pipeline [-h] --run-name RUN_NAME [--seed SEED] [--attribution-map ATTRIBUTION_MAP]
                        [--data-dir-prefix DATA_DIR_PREFIX] [--word-dropout WORD_DROPOUT]
                        [--hyp-only-model HYP_ONLY_MODEL] [--train-hyp] [--hard-validation]
                        [--test-with-prior] [--model-name MODEL_NAME] [--weight-decay WEIGHT_DECAY]

optional arguments:
  -h, --help            show this help message and exit
  --run-name RUN_NAME, -n RUN_NAME
                        Name of run and output file
  --seed SEED, -s SEED  Random seed
  --attribution-map ATTRIBUTION_MAP, -am ATTRIBUTION_MAP
                        path of attribution maps folder
  --data-dir-prefix DATA_DIR_PREFIX
                        Prefix of the path to data
  --word-dropout WORD_DROPOUT, -wdo WORD_DROPOUT
                        Word dropout rate during training
  --hyp-only-model HYP_ONLY_MODEL, -hom HYP_ONLY_MODEL
                        If you want to weigh loss by htpothesis only output
  --train-hyp
  --hard-validation, -hv
  --test-with-prior, -twp
  --model-name MODEL_NAME
                        Name of the huggingface model
  --weight-decay WEIGHT_DECAY, -wd WEIGHT_DECAY
```

**Generate**:

```
usage: main.py generate [-h] [--data-dir-prefix DATA_DIR_PREFIX] --model-path MODEL_PATH
                        [--model-type MODEL_TYPE] [--model-name MODEL_NAME] [--bs-test BS_TEST]
                        [--save-results SAVE_RESULTS] [--generate-all-labels]

optional arguments:
  -h, --help            show this help message and exit
  --data-dir-prefix DATA_DIR_PREFIX
                        Prefix of the path to data
  --model-path MODEL_PATH, -mp MODEL_PATH
                        Path of the first model
  --model-type MODEL_TYPE, -mt MODEL_TYPE
                        Type of the first model
  --model-name MODEL_NAME, -mn MODEL_NAME
                        Name of the first model
  --bs-test BS_TEST, -bst BS_TEST
                        Test batch size
  --save-results SAVE_RESULTS, -sr SAVE_RESULTS
                        Pass path if you want to save the results
  --generate-all-labels, -gal
                        Generate premises for all the labels and not just for gold labels
```
