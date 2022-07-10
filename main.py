import os
import copy
import yaml
import random
import argparse
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torchsummary import summary
from tqdm import tqdm
from prettytable import PrettyTable
from scipy.io import savemat,loadmat
from sklearn.model_selection import StratifiedKFold
from utils.train_test_algo import *
from utils.utils import *
from utils.eeg_models import ShallowConvNet, DeepConvNet, EEGNet, DeepConvNet_ERN

parser = argparse.ArgumentParser()

parser.add_argument('--subject', '-s', type=int, nargs='+', default=[1])
parser.add_argument('--defense_method', '-dm', type=str, default='none')
parser.add_argument('--attack_method', '-am', type=str, default='fgsm, tlm_uap, sap')
parser.add_argument('--debug', '-d', action="store_true", default=False)
parser.add_argument('--model', '-m', type=str, default='e')
parser.add_argument('--data', type=str, default='')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dataset', '-ds', type=str, default='bci42a')
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--sall', action="store_true", default=False)
parser.add_argument('--no_train','-nt', action="store_true", default=False)
parser.add_argument('--parameter_adjustment', '-pa', action="store_true", default=False)

parser.add_argument('--dropout_rate','-dr', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
parser.add_argument('--n_repeats', type=int, default=1)
parser.add_argument('--n_splits', type=int, default=5)
parser.add_argument('--epochs1', type=int, default=1500)
parser.add_argument('--epochs2', type=int, default=250)
parser.add_argument('--random_seed', '-rs', type=int, default=20211025)

parser.add_argument('--alpha', type=float, default=0.5)

parser.add_argument('-f')

# load config.yaml
curPath = os.path.dirname(os.path.realpath(__file__))
yamlPath = os.path.join(curPath, "config.yaml")
f = open(yamlPath, 'r', encoding='utf-8')
cfg = f.read()
config_dict = yaml.safe_load(cfg)

def main(args):
    start_time = datetime.datetime.now()
    log_time_s = str(start_time)[5:19].replace(' ','#').replace(':','-')
    print('start time:',start_time)

    df_method = args.defense_method
    attack_type = args.attack_method
    subjects = args.subject
    debug = args.debug
    data_type = args.data
    model_dict = {'e':'eegnet', 'd':'deep', 's':'shallow'}
    model_type = args.model
    model_name = model_dict[model_type]
    dataset_type = args.dataset

    dropoutP = args.dropout_rate
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    n_repeats = args.n_repeats            #repeat time
    n_splits = args.n_splits              #cross validation fold for within-subject
    epochs1 = args.epochs1
    epochs2 = args.epochs2
    random_seed = args.random_seed
    tag = args.tag
    no_train = args.no_train

    is_sall = args.sall

    random_num = random.randint(1,99)
    print('random number:',random_num)

    torch.manual_seed(random_seed) 
    kf = StratifiedKFold(n_splits=n_splits)

    eps_list = config_dict[dataset_type]['eps_list']
    epsilon = config_dict[dataset_type]['epsilon']
    step_size = config_dict[dataset_type]['epsilon']

    if df_method in ['trades', 'mart']:
        beta = config_dict[dataset_type]['beta'][model_type][df_method]
    else:
        beta = 0
   
    if is_sall:
        n_sub = config_dict[dataset_type]['n_sub']
        subjects = [n+1 for n in range(n_sub)]
        if dataset_type == 'p300':
            subjects.remove(5)
        if dataset_type == 'ern':
            subjects.remove(2)
            subjects.remove(3)
            subjects.remove(12)
            subjects.remove(13)
    
    print('subject:',subjects)

    # hrule styles
    FRAME = 0
    ALL   = 1
    NONE  = 2
    HEADER = 3
    my_table = PrettyTable()
    my_table.hrules=ALL
    my_table.header=False
    my_table.add_row(["model", "device", "attack method", "defense method"])
    my_table.add_row([model_name, "{}:{}".format(device, gpu), attack_type, df_method])
    my_table.add_row(["dataset", "epsilon", "step_size", "subjects"])
    my_table.add_row([dataset_type, epsilon, step_size, subjects if not is_sall else "1-"+str(n_sub)])
    my_table.add_row(["batch size", "dropout rate", "lr rate", "beta"])
    my_table.add_row([batch_size, dropoutP, learning_rate, beta])


    if debug:
        print('-------------DEBUG MODE-------------')
        # X = X[:,:,[7,9,11]]
        n_repeats = 2
        betas = [0]
        epochs1 = 2
        epochs2 = 2

    result_folder_path = os.path.join(os.path.dirname(__file__), 'output/result', dataset_type, model_type)
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    result_save_path = os.path.join(result_folder_path, 
                                    'cv_white_'+str(df_method)+
                                    '#'+log_time_s+'#'+tag+'.mat')
    if args.parameter_adjustment:
        result_save_path = os.path.join(result_folder_path, 
                                    'cv_white_'+str(df_method)+
                                    '#'+log_time_s+'_pa_'+'#'+tag+'.mat')

    FINISH_INIT = False

    #for every subject
    sub_bar = tqdm(total = len(subjects), position=0, dynamic_ncols=True, leave=False)
    for s_idx in range(len(subjects)):
        tqdm.write(my_table.get_string(title="White Box Within Subject"))
        test_subject = subjects[s_idx]

        sub_bar.set_description(desc="subject:   " + str(test_subject))

        # load dateset
        data_abspath = os.path.join(os.path.dirname(__file__), 'data')

        if dataset_type == 'bci42a':
            dataset = loadmat(os.path.join(data_abspath, 'bci42a/sall_bp_ems_bci42a.mat'))
            X = dataset['data'][test_subject-1][:288]
            y = dataset['label'][test_subject-1][:288]
            class_weight = [1, 1, 1, 1]
            class_weight = torch.tensor(class_weight, dtype=torch.float).to(device)
     
        if dataset_type == 'ern':
            dataset = loadmat(os.path.join(data_abspath,'ern/sall_bp_z.mat'))
            X = dataset['data'][test_subject-1].astype('float32')
            y = dataset['label'][test_subject-1]
            num_0 = y.tolist().count(0)
            num_1 = y.tolist().count(1)
            class_weight = [num_0/y.shape[0], num_1/y.shape[0]]   
            class_weight = torch.tensor(class_weight, dtype=torch.float).to(device)

        if dataset_type == 'p300':
            dataset = loadmat(os.path.join(data_abspath,'p300/s'+str(test_subject)+'_bp_z.mat'))
            X = dataset['data'].astype('float32')
            y = dataset['label'].flatten()

            target = np.where(y == 1)[0].tolist()
            non_target = np.where(y == 0)[0].tolist()
            random.seed(random_seed)
            non_target = random.sample(non_target, len(target))
            X = X[target + non_target]
            y = y[target + non_target]

            num_0 = y.tolist().count(0)
            num_1 = y.tolist().count(1)
            class_weight = [num_0/y.shape[0], num_1/y.shape[0]] 
            class_weight = torch.tensor(class_weight, dtype=torch.float).to(device)

        tqdm.write('class weight:'+str(class_weight))

        chans, samples = X.shape[-2], X.shape[-1]
        num_classes = len(np.unique(y))

        if not FINISH_INIT:
            total_result = np.zeros((len(subjects), 1+3*len(eps_list), n_repeats))
            FINISH_INIT = True

        _train_index, _test_index = [], []
        for train_idx, test_idx in kf.split(X, y):
            _train_index.append(train_idx.tolist())
            _test_index.append(test_idx.tolist()) 
        
        X = torch.from_numpy(X).to(device)
        y = torch.LongTensor(y).to(device)

        total_acc = np.zeros((3*len(eps_list)+1, n_splits))

        repeat_time = -1
        # repeat n times
        for r_idx in tqdm(range(n_repeats),  desc='repeat times', position=1, dynamic_ncols=True, leave=False):
            train_index, test_index = copy.deepcopy(_train_index), copy.deepcopy(_test_index)
            repeat_time += 1
            fold = -1
            # k-fold cross validation
            for j in tqdm(range(n_splits), desc='cv fold.....',position=2, dynamic_ncols=True, leave=False):        
                fold = fold + 1

                if args.parameter_adjustment:
                    model_folder_path = os.path.join(os.path.dirname(__file__), 'output/model/benign', dataset_type, 'pa', df_method)
                    if not os.path.exists(model_folder_path):
                        os.makedirs(model_folder_path)
                    model_path = os.path.join(model_folder_path,
                        'cv_white_s'+str(test_subject)+'_'+str(repeat_time+1)+'_'+str(fold+1)+
                        '#'+tag+'_'+str(random_num)+'.pth')
                else:
                    model_folder_path = os.path.join(os.path.dirname(__file__), 'output/model', dataset_type, model_name, df_method)
                    if not os.path.exists(model_folder_path):
                        os.makedirs(model_folder_path)
                    model_path = os.path.join(model_folder_path, 
                            'cv_white_s'+str(test_subject)+'_'+str(repeat_time+1)+'_'+str(fold+1)+
                            '#'+tag+'.pth')

                X_train_valid, y_train_valid = X[train_index[j]], y[train_index[j]] 

                test_idx = test_index[j]
                valid_idx = test_index[j+1] if j<n_splits-1 else test_index[0]
                for v_idx in valid_idx:
                    train_index[j].remove(v_idx)
                train_idx = train_index[j]

                X_test, y_test = X[test_idx], y[test_idx]
                X_valid, y_valid = X[valid_idx], y[valid_idx]
                X_train, y_train = X[train_idx], y[train_idx]

                X_train = X_train[:,0:1]
                X_valid = X_valid[:,0:1]
                X_train_valid = X_train_valid[:,0:1]

                X_train, y_train = shuffle_data(X_train, y_train, random_seed)
                X_train_valid, y_train_valid = shuffle_data(X_train_valid, y_train_valid, random_seed)
                
                tqdm.write('X_train shape:'+str(X_train.shape))


                if (dataset_type == 'p300' or dataset_type == 'ern') and model_type == 'd':
                    model = DeepConvNet_ERN(nChan=chans, nTime=samples, nClass=num_classes, dropoutP=dropoutP).to(device)
                elif model_type == 'e':  
                    model = EEGNet(nChan=chans, nTime=samples, nClass=num_classes, dropoutP=dropoutP).to(device)
                elif model_type == 'd':
                    model = DeepConvNet(nChan=chans, nTime=samples, nClass=num_classes, dropoutP=dropoutP).to(device)
                elif model_type == 's':
                    model = ShallowConvNet(nChan=chans, nTime=samples, nClass=num_classes, dropoutP=dropoutP).to(device)
                else:
                    raise ValueError('Model did not exist, please choose correct model.')

            
                loss_fn = nn.CrossEntropyLoss().to(device)
                # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-07)

                # print(summary(model,(1, chans, samples)))

                if not no_train:
                    #training stage 1
                    best_acc = -1.0
                    best_loss = float('inf')

                    for _ in tqdm(range(epochs1), position=4, desc='training 1/2', dynamic_ncols=True, colour='green', leave=False):
                        train_acc, train_loss = train(model, device, loss_fn, optimizer, class_weight=class_weight,
                                                        data=X_train, label=y_train, batch_size=batch_size,
                                                        df_method=df_method,            
                                                        step_size=step_size,
                                                        epsilon=epsilon, 
                                                        config_dict=config_dict[dataset_type]['beta'][model_type],                            
                                                        )
                        valid_acc, valid_loss = test(model, device, loss_fn, data=X_valid, label=y_valid, batch_size=batch_size,)
                        if train_loss < best_loss:
                            best_loss = train_loss
                        if valid_acc > best_acc:
                            best_acc = valid_acc
                            es = 0
                            torch.save(model.state_dict(),model_path)
                        else: 
                            es += 1
                            if es >= 160:
                                break # early stopping

                    #training stage 2
                    best_loss_2 = float('inf')
                    state_dict = torch.load(model_path)
                    model.load_state_dict(state_dict)
                    for _ in tqdm(range(epochs2), position=4, desc='training 2/2', dynamic_ncols=True, colour='blue', leave=False):
                        train_acc, train_loss = train(model, device, loss_fn, optimizer, class_weight=class_weight,
                                                        data=X_train_valid, label=y_train_valid, batch_size=batch_size,
                                                        df_method=df_method, 
                                                        step_size=step_size,
                                                        epsilon=epsilon,
                                                        config_dict=config_dict[dataset_type]['beta'][model_type],
                                                        )
                        valid_acc, valid_loss = test(model, device, loss_fn, data=X_test, label=y_test, batch_size=batch_size)
                        if valid_loss < best_loss:
                            torch.save(model.state_dict(),model_path)
                            break
                        if valid_loss < best_loss_2:
                            best_loss_2 = valid_loss
                            torch.save(model.state_dict(),model_path)

                # print("load the best model...")
                state_dict = torch.load(model_path)
                model.load_state_dict(state_dict) 

                acc, _ = test(model, device, loss_fn, data=X_test, label=y_test, batch_size=batch_size)
                total_acc[0, fold] = acc
                tqdm.write(f"Test_benign[{fold+1:>d}/{n_splits:>d}]: acc: {(100*acc):>4.1f}%")

                offset = 1
                for eps_idx in range(len(eps_list)):
                    eps = eps_list[eps_idx]
                    acc = test_fgsm(model, device, data=X_test, label=y_test, batch_size=batch_size, eps=eps)
                    total_acc[eps_idx+offset, fold] = acc
                    tqdm.write(f"Test_fgsm[{fold+1:>d}/{n_splits:>d}]: acc: {(100*acc):>4.1f}%, eps: {eps:>.5f}")
                if not args.parameter_adjustment:
                    offset = len(eps_list) + 1
                    for eps_idx in range(len(eps_list)):
                        eps = eps_list[eps_idx]
                        if eps_idx == 0:
                            acc_last = total_acc[0, fold]
                        else:
                            acc_last = acc
                        v, acc = test_tlm_uap(model=model, device=device, X_train=X_train_valid, y_train=y_train_valid,
                                            X_test=X_test, y_test=y_test, batch_size=batch_size, eps=eps, acc_last=acc_last, class_weight=class_weight)
        
                        v = torch.from_numpy(v).float().to(device)                       
                        acc, _ = test(model, device, loss_fn, data=X_test+v, label=y_test, batch_size=batch_size)
                        total_acc[eps_idx + offset, fold] = acc
                        tqdm.write(f"Test_tlm_uap[{fold+1:>d}/{n_splits:>d}]: acc: {(100*acc):>4.1f}%, eps: {eps:>.5f}")

                    offset = 2*len(eps_list)+1
                    for eps_idx in range(len(eps_list)):
                        eps = eps_list[eps_idx]
                        acc = test_sap(model, device, data=X_test, label=y_test, batch_size=batch_size, eps=eps)
                        total_acc[eps_idx+offset, fold] = acc
                        tqdm.write(f"Test_sap[{fold+1:>d}/{n_splits:>d}]: acc: {(100*acc):>4.1f}%, eps: {eps:>.5f}")
              
            ave_acc = np.mean(total_acc, axis=1)
            tqdm.write(f"round:{r_idx+1:>d}/{n_repeats:>d},the acc is:{ave_acc}, avg:{np.mean(ave_acc[:len(eps_list)+1])}") 

            total_result[s_idx, :, r_idx] = ave_acc
    
        tqdm.write(f"Average results of {n_repeats} times, the acc is: {np.mean(total_result[s_idx], axis=1)}, avg:{np.mean(np.mean(total_result[s_idx], axis=1)[:len(eps_list)+1])}")
        savemat(result_save_path,{'subjects':subjects, 'lr':learning_rate,'df_method':str(df_method),
            'beta':betas,'epsilon':eps_list,'acc':total_result, 'data_type':data_type, 'model_type':model_type})
        tqdm.write('Subject'+str(test_subject)+' result has been saved.')
        sub_bar.update()

if __name__ == "__main__":

    args = parser.parse_args()

    gpu = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using {}({}) device".format(device, gpu))

    np.set_printoptions(linewidth=100)

    main(args)
