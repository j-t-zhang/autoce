import os
from tqdm import tqdm
from multiprocessing import Process
import pandas as pd
import pickle
import math
# Easily call all methods to get results


import argparse
parser = argparse.ArgumentParser(description='RUN')
# parser.add_argument('--method', type=str, help='method', default='deepdb')
parser.add_argument('--num1', type=int, help='num_st', default='1240')
parser.add_argument('--num2', type=int, help='num_ed', default='1260')
args = parser.parse_args()

num1 = args.num1
num2 = args.num2

# Preprocess
def run_preprocess(version):
    # version = f'table{2}'

    os.chdir('./learnedcardinalities/mscn')
    pretrain = 'python preprocessing.py --datasets-dir ../../../data/benchmark/ --raw-query-file ../../../data/benchmark/' + version + 'train.sql' + ' --min-max-file ../data/' + version + '_min_max_vals.csv ' + '--table ' + version + ' --alias Syn'
    pretest = 'python preprocessing.py --datasets-dir ../../../data/benchmark/ --raw-query-file ../../../data/benchmark/' + version + 'test.sql' + ' --min-max-file ../data/' + version + '_min_max_vals.csv ' + '--table ' + version + ' --alias Syn'
    os.system(pretrain)
    os.system(pretest)
    os.chdir('../..')

# Bayescard
def run_bayescard(version,sample_size):
    # version = f'table{3}'

    os.chdir('./bayescard')
    os.system('python run_experiment.py --dataset general --generate_models --csv_path ../../data/benchmark/' + version + '.csv --model_path Benchmark/General --learning_algo chow-liu --max_parents 1 --sample_size '+ str(sample_size) + ' --version ' + version)
    os.system('python run_experiment.py --dataset general --evaluate_cardinalities --csv_path ../../data/benchmark/' + version + '.csv --model_path Benchmark/General/' + version[len(version)-4:len(version)] + '.csvchow-liu_1.pkl --query_file_location ../../data/benchmark/' + version + 'test.sql --infer_algo exact-jit --version ' + version) 
    os.chdir('..')

# DeepDB
# modify database_name 
def run_deepdb(version,sample_size):
    # version = f'table{1}'

    # true_cardinalities.csv
    os.chdir('./deepdb/deepdb_run/')
    path = "../../../data/benchmark/"
    sql_path = path + version + "test.sql"
    sql_path2 = '../benchmark/model_truecard/' + f'true_cardinalities{version}.csv'  # true_cardinalities 
    f2 = open(sql_path2, 'w')
    f2.write('query_no,query,cardinality_true\n')
    i = 0
    with open(sql_path, 'r') as f:
        for line in f.readlines():
            strt = line[len(line)-10: len(line)]
            tmpindex = strt.index(',')
            strt = strt[tmpindex+1: len(strt)]
            tmpz = str(i) + ',' + str(i+1) + ',' + strt
            f2.write(tmpz)
            i+=1
    f2.close()

    # ../../../data/benchmark/table_nohead
    pre = 'python3 maqp.py --generate_hdf --generate_sampled_hdfs --generate_ensemble --csv_path ../../../data/benchmark/table_nohead --ensemble_path ../benchmark/model_truecard/ --version ' + version  + ' --per_sample_size ' + str(sample_size)
    run = 'python3 maqp.py --evaluate_cardinalities --rdc_spn_selection --max_variants 1 --pairwise_rdc_path ../benchmark/spn_ensembles/pairwise_rdc.pkl --dataset general ' + '--target_path ../benchmark/model_truecard/' + version + 'test.sql.deepdb.results.csv ' + '--ensemble_location ../benchmark/model_truecard/' + version + '.sql.deepdb.model.pkl ' + '--query_file_location ../../../data/benchmark/' + version + 'test.sql ' + f'--ground_truth_file_location ../benchmark/model_truecard/true_cardinalities{version}.csv --version ' + version  + ' --per_sample_size ' + str(sample_size)

    os.system(pre)
    os.system(run)
    os.chdir('../..')

# Naru
def run_naru(version,sample_rate):
    # version = f'table{2}'

    os.chdir('./naru')
    os.system('python train_model.py --version '  + version + f' --num-gpus=1 --dataset=dmv --epochs=100 --warmups=8000 --bs=2048 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --sample_rate {sample_rate}')
    os.system('python eval_model.py --testfilepath ../../data/benchmark/ --version '  + version + ' --table '  + version + f' --alias Syn --dataset=dmv --glob=\'<ckpt from above>\' --num-queries=1000 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --sample_rate {sample_rate}')
    os.chdir('..')

# UAE
def run_uae(version,sample_rate):
    # version = f'table{2}'

    os.chdir('./uae')
    os.system(f'python train_uae.py --num-gpus=1 --cuda-num 2 --dataset=dmv2 --epochs=50 --constant-lr=5e-4 --bs=2048  --residual --layers=2 --fc-hiddens=128 --direct-io --column-masking --version {version} --sample_rate {sample_rate}')
    
    os.system(f'python eval_model.py --dataset=dmv2 --glob=\'<ckpt from above>\' --psample=1000 --residual --direct-io --column-masking --version {version} --sample_rate {sample_rate}')
    os.chdir('..')

# UAE2
def run_uae2(version,sample_rate):
    # version = f'table{2}'

    os.chdir('./uae')
    os.system(f'python train_uae.py --num-gpus=1 --cuda-num 1 --dataset=dmv2 --epochs=50 --constant-lr=5e-4 --bs=2048  --residual --layers=2 --fc-hiddens=128 --direct-io --column-masking --version {version} --sample_rate {sample_rate}')
    
    os.system(f'python eval_model.py --dataset=dmv2 --glob=\'<ckpt from above>\' --psample=1000 --residual --direct-io --column-masking --version {version} --sample_rate {sample_rate}')
    os.chdir('..')

# Mscn
# 修改 --queries 150
def run_mscn(version):
    # version = f'table{0}'

    os.chdir('./learnedcardinalities')
    train = 'python train.py --min-max-file data/' + version + '_min_max_vals.csv --queries 500 --epochs 45 --batch 256 --hid 256 --train-query-file workload/' + version + 'train.sql --test-query-file workload/' + version + 'test.sql --train --version ' + version
    test = 'python train.py --min-max-file data/' + version + '_min_max_vals.csv --queries 500 --epochs 45 --batch 256 --hid 256 --train-query-file workload/' + version + 'train.sql --test-query-file workload/' + version + 'test.sql --version ' + version
    os.system(train)
    os.system(test)
    os.chdir('..')

def run_xgb(version):
    # version = f'table{0}'

    os.chdir('./xgboost_localnn')
    os.system('python run.py --train-file ../learnedcardinalities/workload/' + version + 'train.sql' + ' --test-file ../learnedcardinalities/workload/' + version + 'test.sql' + ' --min-max-file ../learnedcardinalities/data/' + version + '_min_max_vals.csv ' + '--model xgb' + ' --version ' + version)
    os.chdir('..')

def run_nn(version):
    # version = f'table{0}'

    os.chdir('./xgboost_localnn')
    os.system('python run.py --train-file ../learnedcardinalities/workload/' + version + 'train.sql' + ' --test-file ../learnedcardinalities/workload/' + version + 'test.sql' + ' --min-max-file ../learnedcardinalities/data/' + version + '_min_max_vals.csv ' + '--model nn' + ' --version ' + version)
    os.chdir('..')


# RUN
def run_all(method,num1,num2):
    f_table_num = open('../data/feature/table_num.list','rb')
    table_num = pickle.load(f_table_num)

    for tablei in tqdm(range(num1,num2)):
        if table_num[tablei]>3:
            table_num[tablei] -= 1
        version = f'table{tablei}'
        data = pd.read_csv(f'../data/benchmark/table{tablei}.csv')
        len_data = len(data)

        sample_rate = math.pow(0.5,table_num[tablei])
        sample_size = sample_rate*len_data

        if method == 'preprocess':
            run_preprocess(version)
        elif method == 'deepdb':
            run_deepdb(version,int(sample_size/4.1))
        elif method == 'naru':
            run_naru(version,round(sample_rate*math.pow(1.8,table_num[tablei]),4))
        elif method == 'mscn':
            run_mscn(version)
        elif method == 'bayescard':
            samples = [5000,4200,3500,2400,2000]
            # print('sample_size:',sample_size)
            run_bayescard(version,samples[table_num[tablei]])# int(sample_size*round(sample_rate*math.pow(1.1,table_num[tablei]),4)))# int(sample_size*math.pow(1.15,table_num[tablei])))
        elif method == 'xgb':
            run_xgb(version)
        elif method == 'nn':
            run_nn(version)
        elif method == 'uae':
            run_uae(version,round(sample_rate*math.pow(1.7,table_num[tablei]),4))
        elif method == 'uae2':
            run_uae2(version,round(sample_rate*math.pow(1.7,table_num[tablei]),4))
    print(method+f' from {num1} to {num2} Done')
    return 



if __name__ == '__main__':

    for me in ['bayescard','deepdb','mscn','naru','nn','xgb','uae']:
        if not os.path.exists(f'./Metric_res/{me}.csv'):
            header0 = 'Index,Qerror_mean,Qerror_50,Qerror_90,Qerror_95,Qerror_99,Qerror_100,Train_time,Evalu_time\n'
            f = open(f'./Metric_res/{me}.csv','w')
            f.write(header0)
            f.close()

    pre1 = Process(target=run_all, args=('preprocess',num1,int(num2/4),))
    pre2 = Process(target=run_all, args=('preprocess',int(num2/4),int(num2*2/4),))
    pre3 = Process(target=run_all, args=('preprocess',int(num2*2/4),int(num2*3/4),))
    pre4 = Process(target=run_all, args=('preprocess',int(num2*3/4),num2,))

    #pre1.start()
    #pre2.start()
    #pre3.start()
    #pre4.start()


    p1 = Process(target=run_all, args=('bayescard',num1,num2,))

    p2 = Process(target=run_all, args=('deepdb',num1,num2,))

    p3 = Process(target=run_all, args=('mscn',num1,int(num2),))
    # p4 = Process(target=run_all, args=('mscn',int(num2/2),num2,))

    p7 = Process(target=run_all, args=('naru',num1,num2,))

    p8 = Process(target=run_all, args=('nn',num1,num2,))

    p9 = Process(target=run_all, args=('xgb',num1,num2,))

    p10 = Process(target=run_all, args=('uae',num1,int(num2),))
    # p11 = Process(target=run_all, args=('uae2',int(num2/2),num2,))
    
    
    # p1.start()
    # p2.start()
    # p3.start()
    # p4.start()

    # p7.start()
    # p8.start()
    # p9.start()
    p10.start()
    # p11.start()
