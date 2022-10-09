import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from dgl.data import GINDataset
from dataloader import GINDataLoader
from ginparser import Parser
from gin import GIN
from utils import preprocess,lable2cate
from dataset import MyDataset
import pickle


def train(args, net, trainloader, optimizer, criterion, epoch):
    net.train()

    running_loss = 0
    total_iters = len(trainloader)
    # setup the offset to avoid the overlap with mouse cursor
    bar = tqdm(range(total_iters), unit='batch', position=2, file=sys.stdout)

    for pos, (graphs, labels) in zip(bar, trainloader):
        # batch graphs will be shipped to device in forward part of model
        labels = labels.to(args.device)
        graphs = graphs.to(args.device)
        feat = graphs.ndata.pop('attr')
        feat = torch.tensor(feat, dtype=torch.float32)  # modify
        # print('feat:',feat)
        # feat = feat.to(args.device)
        # print('trainloader:',trainloader)
        # print('graphs:',graphs)
        
        # print('feat:',feat.shape)
        outputs = net(graphs, feat)
        
        # print('outputs:',outputs)
        # categ = lable2cate(labels)
        labels = torch.where(torch.isnan(labels), torch.full_like(labels, 0.5), labels)
        # outputs = torch.where(torch.isnan(outputs), torch.full_like(outputs, 0), outputs)
        # print('labels:',labels)
        loss = criterion(outputs.float(), labels.float())
        # loss = outputs.sum() # criterion(outputs, labels)
        # print('loss:',loss)
        # print(loss.grad)
        # print('MSELoss:',loss)
        
        running_loss += loss.item()
        # print('running_loss:',loss)
        # loss= torch.tensor(loss, dtype=torch.float32).requires_grad_()  # modify

        # backprop
        optimizer.zero_grad()
        
        loss.backward()
        # outputs.sum().backward()
        # for name, parms in net.named_parameters(): 
            # print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', parms.grad) # torch.mean(parms.grad)
        optimizer.step()
        

        # report
        bar.set_description('epoch-{}'.format(epoch))
    bar.close()
    # the final batch will be aligned
    running_loss = running_loss / total_iters

    return running_loss


def eval_net(args, net, dataloader, criterion):
    net.eval()

    total = 0
    total_loss = 0
    err_all = 0
    err_bad_all = 0
    predicteds = torch.tensor([])
    predicteds = predicteds.to(args.device)
    total_correct,total_correct2,total_correct3,total_correct4,total_correct5,total_correct6,total_correct7 = 0,0,0,0,0,0,0

    for data in dataloader:
        
        graphs, labels = data
        graphs = graphs.to(args.device)
        labels = labels.to(args.device)
        feat = graphs.ndata.pop('attr')
        feat= torch.tensor(feat, dtype=torch.float32)  # modify
        total += len(labels)
        outputs = net(graphs, feat)
        labels = torch.where(torch.isnan(labels), torch.full_like(labels, 0.5), labels)
        _, predicted = torch.max(outputs.data, 1)
        # _, min_predicted = torch.min(outputs.data, 1)
        torch.cat
        predicteds = torch.cat((predicteds,predicted))
        # print('predicted:',list(predicted))
        _, catelabel = torch.max(labels.data, 1)
        catelabel2 = np.argsort(torch.Tensor.cpu(labels.data))[:,-2]
        catelabel3 = np.argsort(torch.Tensor.cpu(labels.data))[:,-3]
        catelabel4 = np.argsort(torch.Tensor.cpu(labels.data))[:,-4]
        catelabel5 = np.argsort(torch.Tensor.cpu(labels.data))[:,-5]
        catelabel6 = np.argsort(torch.Tensor.cpu(labels.data))[:,-6]
        catelabel7 = np.argsort(torch.Tensor.cpu(labels.data))[:,-7]

        total_correct += (predicted == catelabel.data).sum().item()
        total_correct2 += (predicted == catelabel2.data.to(args.device)).sum().item()
        total_correct3 += (predicted == catelabel3.data.to(args.device)).sum().item()
        total_correct4 += (predicted == catelabel4.data.to(args.device)).sum().item()
        total_correct5 += (predicted == catelabel5.data.to(args.device)).sum().item()
        total_correct6 += (predicted == catelabel6.data.to(args.device)).sum().item()
        total_correct7 += (predicted == catelabel7.data.to(args.device)).sum().item()
        # print('catelabel:',catelabel)
        # print('predicted:',predicted)
        # print('labels:',labels)
        errs = (labels[[id for id in range(len(catelabel))],catelabel] - labels[[id for id in range(len(predicted))],predicted]) / labels[[id for id in range(len(predicted))],predicted]

        errs_bad = (labels[[id for id in range(len(predicted))],predicted] - labels[[id for id in range(len(catelabel7))],catelabel7]) / labels[[id for id in range(len(predicted))],predicted]
        
        err = errs.sum()
        err_bad = errs_bad.sum()
        # print('err:',err)

        # total_correct += (predicted == catelabel.data).sum().item()
        loss = criterion(outputs, labels)
        # crossentropy(reduce=True) for default
        total_loss += loss.item() * len(labels)
        loss= torch.tensor(loss, dtype=torch.float32).requires_grad_()  # modify
        err_all += err
        err_bad_all += err_bad
        acc_list = [round(1.0*item/total,3) for item in [total_correct,total_correct2,total_correct3,total_correct4,total_correct5,total_correct6,total_correct7]]  
    # print('predicteds:',len(predicteds))
    loss, err, err_bad = 1.0*total_loss / total, 1.0*err_all/total, 1.0*err_bad_all/total 
    err, err_bad = str(round(err.item()*100,3))+'%', str(round(err_bad.item()*100,3))+'%'
    net.train()

    return loss, [err, err_bad], acc_list, predicteds.cpu().numpy()


def main(args, dataset):

    # set up seeds, args.seed supported
    torch.manual_seed(seed=args.seed)
    np.random.seed(seed=args.seed)

    is_cuda = not args.disable_cuda and torch.cuda.is_available()

    if is_cuda:
        args.device = torch.device("cuda:" + str(args.device))
        torch.cuda.manual_seed_all(seed=args.seed)
    else:
        args.device = torch.device("cpu")

    # dataset = GINDataset(args.dataset, not args.learn_eps)
    # print('dataset:',dataset)

    trainloader, validloader = GINDataLoader(
        dataset, batch_size=args.batch_size, device=args.device,
        seed=args.seed, shuffle=True,
        split_name='rand', split_ratio=0.75).train_valid_loader()
    # or split_name='fold10', fold_idx=args.fold_idx  

    model = GIN(
        args.num_layers, args.num_mlp_layers,
        dataset.dim_nfeats, args.hidden_dim, dataset.gclasses,
        args.final_dropout, args.learn_eps,
        args.graph_pooling_type, args.neighbor_pooling_type).to(args.device)

    criterion = torch.nn.MSELoss()  #  nn.CrossEntropyLoss()  # defaul reduce is true
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # it's not cost-effective to hanle the cursor and init 0
    # https://stackoverflow.com/a/23121189
    tbar = tqdm(range(args.epochs), unit="epoch", position=3, ncols=0, file=sys.stdout)
    vbar = tqdm(range(args.epochs), unit="epoch", position=4, ncols=0, file=sys.stdout)
    lrbar = tqdm(range(args.epochs), unit="epoch", position=5, ncols=0, file=sys.stdout)

    for epoch, _, _ in zip(tbar, vbar, lrbar):

        train(args, model, trainloader, optimizer, criterion, epoch)
        scheduler.step()

        '''train_loss, train_acc = eval_net(
            args, model, trainloader, criterion)
        tbar.set_description(
            'train set - average loss: {:.4f}, accuracy: {:.0f}%'
            .format(train_loss, 100. * train_acc))

        valid_loss, valid_acc = eval_net(
            args, model, validloader, criterion)
        vbar.set_description(
            'valid set - average loss: {:.4f}, accuracy: {:.0f}%'
            .format(valid_loss, 100. * valid_acc))'''

        train_loss, train_err, train_acc,_ = eval_net(
            args, model, trainloader, criterion)
        tbar.set_description(
            'train set - average loss: {:.4f}, err: {}, train_acc:{}'
            .format(train_loss, train_err, train_acc))

        valid_loss, valid_err, valid_acc, predicts = eval_net(
            args, model, validloader, criterion)

        if epoch == args.epochs-1:
            f_res = open(f'./res/res{args.acc_wgt}.array','wb')
            pickle.dump(predicts,f_res)

        vbar.set_description(
            'valid set - average loss: {:.4f}, err: {}%, valid_acc:{}'
            .format(valid_loss, valid_err, valid_acc))
        

        if not args.filename == "":
            with open(args.filename, 'a') as f:
                f.write('%s %s %s %s' % (
                    args.dataset,
                    args.learn_eps,
                    args.neighbor_pooling_type,
                    args.graph_pooling_type
                ))
                f.write("\n")
                f.write("%f %f %f %f" % (
                    train_loss,
                    train_acc,
                    valid_loss,
                    valid_acc
                ))
                f.write("\n")

        lrbar.set_description(
            "Learning eps with learn_eps={}: {}".format(
                args.learn_eps, [layer.eps.data.item() for layer in model.ginlayers]))

    tbar.close()
    vbar.close()
    lrbar.close()

    # f_model = open('./model/advisor.model','wb')
    # pickle.dump(model,f_model)


if __name__ == '__main__':
    A_dict,W_dict,feat_dict,label_acc_dict,label_eff_dict = preprocess()
    


    args = Parser(description='GIN').args
    print('show all arguments configuration...')
    print(args)
    dataset = MyDataset(feat_dict,A_dict,W_dict,label_acc_dict,label_eff_dict,acc_wgt=args.acc_wgt)
    main(args, dataset)
