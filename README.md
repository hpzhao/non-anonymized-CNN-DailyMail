A script to produce non-anonymized CNN and DailyMail for summary. **Reference**: [abisee/cnn-daulymail](https://github.com/abisee/cnn-dailymail)

## Environment

+ python 3.6

## Features

+ tokenized by CoreNLP
+ non-anonymized
+ lowercase
+ remove artical infomation
+ multiprocess
+ json(more readable)

## How to use it?

#### 1. Download data

Download the stories directories from [here](https://cs.nyu.edu/~kcho/DMQA/) for both CNN and Daily Mail.

#### 2. Download CoreNLP

Download and unzip CoreNLP from [here](https://stanfordnlp.github.io/CoreNLP/). Add the following command in your bash_profile:  

```bash
export CLASSPATH=$CLASSPATH:/path/to/stanfordnlp-corenlp-full-2018-02-27/stanford-corenlp-3.9.1.jar

```
#### 3. Make dataset

```bash

# for dailymail(similar for cnn)
# if your device has multiple CPUs, you could speed up by setting -worker_num

python make_dataset.py -stories_dir dailymail/stories -tokenized_storeis_dir dailymail/tokenized_storeis -train_urls url_lists/dailymail_wayback_training_urls.txt -test_urls url_lists/dailymail_wayback_test_urls.txt -val_urls dailymail_wayback_validation_urls.txt -output_dir dailymail 

```
