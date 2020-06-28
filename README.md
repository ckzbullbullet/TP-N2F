# TP-N2F
 TP-N2F model
## Paper
 This is Pytorch implementation of TP-N2F model for paper "Mapping Natural-language Problems to Formal-language Solutions Using Structured Neural Representations"

## Requirements
 This project supports both GPUs and CPUs. The following requirements are for GPU users. If you want to train or test the models on CPUs, you could ignore the packages related CUDA. All required packages are in requirements.txt. Both Linux and Windows users can use "pip" to install required packages.
 ```
 Ubuntu 16.04+ or Windows 10
 Python 3.4+
 pytorch 1.4.0
 cuda9+
 cudnn7+
 ```
## Data
We cleaned the data for both dataset. You can download them at [here](https://drive.google.com/file/d/15apXXavs4nmdeZnLUDKNuQcxZXLhULYy/view?usp=sharing).
Download the zip file and unzip it at the same level of src folder. 

## Train models
 You can find the Pytorch implementation of TP-N2F model in 'src/model.py' file. If you want to train models on MathQA and AlgoLisp datasets, you can run the two files 'src/run_tpn2f_mathqa.py' and 'src/run_tpn2f_lisp.py'. The following commands can start the training process (if you train the model on CPU, add '--no_cuda' option for both commands). For the details of hyper-parameters, you can check the specific files and change them for your own preference. You can download the data of both datasets in the next section.
 MathQA
 ```
 python run_tpn2f_mathqa.py --data_dir <data_path> --output_dir <output_path> --do_train --do_eval --train_batch_size <train_bs> --eval_batch_size <eval_bs> --learning_rate <lr> --num_train_epochs <epochs> --binary_rela <True/False> --bidirectional <True/False> --max_seq_length <seqlen> --src_layer <src_ly> --trg_layer <trg_ly> --nSymbols <nfiller> --nRoles <nrole> --dSymbols <dfiller> --dRoles <drole> --temperature <temp> --dOpts <dopt> --dArgs <darg> --dPoss <dpos> --attention <dot/tpr> --sum_T <True/False> --reason_T <1/2> --lr_decay <True/False>
 ```
 AlgoLisp
 ```
 python run_tpn2f_lisp.py --data_dir <data_path> --output_dir <output_path> --do_train --do_eval --train_batch_size <train_bs> --eval_batch_size <eval_bs> --learning_rate <lr> --num_train_epochs <epochs> --clean_data <0/1> --bidirectional <True/False> --max_seq_length <seqlen> --src_layer <src_ly> --trg_layer <trg_ly> --nSymbols <nfiller> --nRoles <nrole> --dSymbols <dfiller> --dRoles <drole> --temperature <temp> --dOpts <dopt> --dArgs <darg> --dPoss <dpos> --attention <dot/tpr> --sum_T <True/False> --reason_T <1/2> --lr_decay <True/False>
 ```

## Duplicate our results
 You can download the data from the link at data section. Once you download the data.zip, you unzip the data folder at the same level of 'src' folder. Then, you can run the following commands under src folder to get the results for both datasets. We created a 'results' folder to save the results (If you want to test them on CPU, you add the option "--no_cuda" for each command).
 MathQA
 ```
 python run_tpn2f_mathqa.py --data_dir ../data/MathQA --output_dir ../results --eval_model_file mathqa_150_50_30_20_10_20_5.model --do_eval --train_batch_size 128 --eval_batch_size 256 --learning_rate 0.001 --num_train_epochs 60 --binary_rela True --bidirectional True
 ```
 AlgoLisp
 ```
 python run_tpn2f_lisp.py --data_dir ../data/AlgoLisp --output_dir ../results --eval_model_file algolisp_150_50_30_30_30_20_5.model --do_eval --train_batch_size 128 --eval_batch_size 256 --learning_rate 0.00115 --num_train_epochs 50 --clean_data 0
 ```
