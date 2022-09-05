# cmna_src

This repository contains the source code for the "Leveraging BERT Encodings for Open-Domain Stance Classification" CMNA paper and can be used for replication of results.

It includes an implementation of the base-BERT model, the novel variants described in the paper and IBM's Project Debater's pro/con stance classifier.

Also included is a script which allows execution and evaluation of each of the models to be customized and ran easily from the terminal.

## Dependencies
Start off by making sure all of the dependencies listed out in the requirements.txt file are installed.
```bash
pip install -r src/requirements.txt
```
Everything needed to run the BERT based models should be included in this file, and upon installation should be ready to run.
Some extra steps are necessary to gain access to the Project Debater's stance classifier's API service.

### PD stance classifier
- Request the API key acess for academic use of the stance classifier at the following address: https://early-access-program.debater.res.ibm.com/academic_use.
- Install the API service as follows:
```bash
wget -P . https://early-access-program.debater.res.ibm.com/sdk/python_api.tar.gz
tar -xvf python_api.tar.gz
cd python_api ; pip install .
rm -f python_api.tar.gz*
```
- Set up an environmental variable named 'DEBATER_API_KEY' with the API key as its value. This can be declared within a conda environment, a jupyter notebook...
Example for a jupyter notebook:
```python
os.environ['DEBATER_API_KEY']='API_key'
```

## Script usage
Start by entering the src folder:
```bash
cd src
```
The main.py script is used to run the system and works as follows:
- The following two flags are obligatory:
    - The -m flag accepts any number of arguments and specifies the model/s that should be trained. 
        - The options are: bert, bert_syn_con, bert_syn_mul, bert_cos_con, bert_cos_mul, bert_cosyn_con, bert_cosyn_mul and pd
    - The -d flag accepts any number of arguments and specifies the dataset/s the model/s should be run against.
        - The options are: pers, multi
- The following flags are optional and allow for control of the execution:
    - The -t flag specifies that the action performed should be to train the specified model/s on the specified dataset/s. (This returns an error if pd is selected)
    - The -e flag specifies that the action performed should be to evaluate the specified model/s against the specified dataset/s and to save the results in the stats folder.
    - The -ep flag specifies the number of epochs to train during the training loop. 
    - The -b flag specifies that the model/s should be run without loading in any fine-tuned weights. 
Any optional flags which are not explicitly declared will be initialized to their default values which can be found in config.yaml.

Example:
If I wanted to train the bert_cosyn_mul model for 10 epochs followed by an evaluation for both the pers and multi datasets, I would write the following:
```bash
python -t -e -m bert_cosyn_mul -d pers multi -ep 10
```
