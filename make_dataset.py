import os
import random
import argparse
import hashlib
import json
from itertools import chain
from multiprocessing import Pool,cpu_count

parser = argparse.ArgumentParser()
parser.add_argument('-stories_dir', default='dailymail/stories')
parser.add_argument('-tokenized_stories_dir', default='dailymail/tokenized_storeis')
parser.add_argument('-train_urls', default='url_lists/dailymail_wayback_training_urls.txt')
parser.add_argument('-test_urls', default='url_lists/dailymail_wayback_test_urls.txt')
parser.add_argument('-val_urls', default='url_lists/dailymail_wayback_validation_urls.txt')
parser.add_argument('-output_dir', default='dailymail/')
parser.add_argument('-worker_num', type=int, default=1)

args = parser.parse_args()

def tokenize_worker(stories):
    mapping = 'pid_%d_rand_%d.map' % (os.getpid(),random.randint(1,1000000))
    with open(mapping,'w') as f:
        for s in stories:
            f.write('%s \t %s\n' % (os.path.join(args.stories_dir,s),os.path.join(args.tokenized_stories_dir,s)))
    command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', mapping]
    os.system(' '.join(command))
    os.remove(mapping)
def tokenize_stories(stories_dir,tokenized_stories_dir):
    print('Start tokenizing stories, it will take some time.')
    
    stories = os.listdir(args.stories_dir)
    group_size = len(stories) // args.worker_num
    groups = []
    for i in range(args.worker_num):
        if i == args.worker_num - 1:
            groups.append(stories[i*group_size : ])
        else:
            groups.append(stories[i*group_size : (i+1)*group_size])

    p = Pool(processes=args.worker_num)
    multi_res = [p.apply_async(tokenize_worker,(s,)) for s in groups]
    res = [res.get() for res in multi_res]
    
def get_hash(urls_list):
    res = []
    for url in urls_list:
        h = hashlib.sha1()
        h.update(url.encode('utf-8'))
        res.append(h.hexdigest()+'.story')
    return res

def read_story(f):
    # Lowercase everything
    lines = [line.strip().lower() for line in open(os.path.join(args.tokenized_stories_dir,f))]
    info = ['by','published :','updated :']
    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    next_is_info = False
    for idx,line in enumerate(lines):
        # empty line or info
        if line == "" or line == '|' or line.startswith('last updated at'): 
            continue
        # artical info
        if line in info:
            next_is_info = True
            continue
        # highlight info
        if line.startswith("@highlight"):
            next_is_highlight = True
            continue
        
        if next_is_highlight:
            highlights.append(line)
        elif next_is_info:
            next_is_info = False
        else:
            article_lines.append(line)

    return {'doc':'\n'.join(article_lines), 'summaries':'\n'.join(highlights)}

def split_worker(fs):
    res = []
    for f in fs:
        res.append(read_story(f))
    return res
def split_dataset(url_file,outfile):
    if args.worker_num == 1 and cpu_count() > 1:
        print('Your device has %d CPUs, you could speed up by setting -work_num' % (cpu_count()))
    url_list = [line.strip() for line in open(url_file)]
    files = get_hash(url_list) 

    group_size = len(files) // args.worker_num
    groups = []
    for i in range(args.worker_num):
        if i == args.worker_num - 1:
            groups.append(files[i*group_size : ])
        else:
            groups.append(files[i*group_size : (i+1)*group_size])

    p = Pool(processes=args.worker_num)
    multi_res = [p.apply_async(split_worker,(fs,)) for fs in groups]
    res = [res.get() for res in multi_res]
    
    with open(outfile,'w',encoding='utf-8') as out_f:
        for row in chain(*res):
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n") 

if __name__ == '__main__':
    if args.worker_num == 1 and cpu_count() > 1:
        print('Your device has %d CPUs, you could speed up by setting -work_num' % (cpu_count()))
    #if not os.path.exists(args.tokenized_stories_dir) : os.makedirs(args.tokenized_stories_dir)
    #tokenize_stories(args.stories_dir, args.tokenized_stories_dir)
    print('Start splitting dataset')
    split_dataset(args.train_urls,os.path.join(args.output_dir,'train.json'))
    split_dataset(args.test_urls,os.path.join(args.output_dir,'test.json'))
    split_dataset(args.val_urls,os.path.join(args.output_dir,'val.json'))
