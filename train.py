import config

import torch,argparse,os,logging,time
import torch.optim as optim
from torch.utils.data import DataLoader

from net import loss
from tools.data_io import tocuda


def train(args, train_args, train_databatch, loss_criterion):
    # creat model, loss, optimizer
    model = config.model
    optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': train_args.lr,}], lr=train_args.lr,)

    # load breakpoint
    start_epoch = train_args.start_epoch
    if args.pre_model is not None:
        checkpoint = torch.load(args.pre_model, )   # map_location=torch.device("cpu")  map_location=DEVICE
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"])

    # load to device
    if train_args.parallel:
        model = torch.nn.DataParallel(model).cuda()
        loss_criterion = loss_criterion.cuda()
    else:
        model.to(train_args.DEVICE)
        loss_criterion = loss_criterion.to(train_args.DEVICE)

    # train
    model.train()
    for epoch in range(start_epoch, train_args.max_epoch+1):
        optimizer.param_groups[0]['lr'] = train_args.lr * ((1 - (epoch-1)/ train_args.max_epoch)**train_args.factor)
        epoch_loss = 0.0
        for batch, data_batch in enumerate(train_databatch):

            data_batch = tocuda(data_batch, train_args.DEVICE, train_args.parallel)
            start_time = time.time()

            outputs = model(data_batch["imgs"], data_batch["extrinsics"], data_batch["intrinsics"], data_batch["depth_range"])
            loss = loss_criterion(outputs, data_batch["ref_depths"], data_batch["depth_range"],)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_loss = loss.detach().item()
            epoch_loss += current_loss
            print("\r"+"epoch: "+str(epoch)+ " batch: "+str(batch + 1)+ "/"+ str(len(train_databatch))
                         + " time:{: .3f}".format(time.time() - start_time)+ " loss:{: .5f}\t".format(current_loss), end="", flush=True)

        logging.info("epoch: "+str(epoch)+" loss:"+str(epoch_loss/len(train_databatch)))

        # save epoch loss
        with open(os.path.join(train_args.pth_path, "epoch_loss.txt"), "a") as f:
            f.write(str(epoch_loss/len(train_databatch)) + "\n")

        # save model
        if epoch % 1 == 0:
            checkpoint = {
                'epoch': epoch,
                'model': None,
                }
            if train_args.parallel:
                checkpoint['model'] = model.module.state_dict()
            else:
                checkpoint['model'] = model.state_dict()
            torch.save(checkpoint, os.path.join(train_args.pth_path, args.dataset+"_" + str(epoch) + ".pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DTU train parameter setting')
    parser.add_argument('-p', '--pre_model', default=None, type=str, help='Pre training model or last model')
    parser.add_argument('-d', '--dataset', default='dtu', type=str, choices=['dtu', 'blendedmvs'], help='Set dataset')
    parser.add_argument('-l', '--cmd_label', default="", type=str, help='show train condition in ps -aux')

    args = parser.parse_args()
    logging.info(args)

    train_args, train_dataset = None, None
    if args.dataset == "dtu":
        load_args = config.LoadDTU()
        train_args = config.TrainArgs()
        # load dataset
        from load.dtutrain import LoadDataset
        train_dataset = LoadDataset(datasetpath=load_args.train_root,
                                    pairpath=load_args.train_pair,
                                    scencelist=load_args.train_label,
                                    lighting_label=load_args.train_lighting_label,
                                    nviews=train_args.nviews,
                                    robust_train=train_args.robust)
    elif args.dataset == "blendedmvs":
        load_args = config.LoadBlendedMVS()
        train_args = config.BlendedMVSArgs()
        # load dataset
        from load.blendedtrain import LoadDataset
        train_dataset = LoadDataset(datasetpath=load_args.train_root,
                                    nviews=train_args.nviews,
                                    robust_train=train_args.robust)
    else:
        print("Error dataset")
        exit()

    # load dataet
    train_databatch = DataLoader(train_dataset, batch_size=train_args.batch_size, shuffle=True,
                                 num_workers=train_args.nworks,
                                 drop_last=True, pin_memory=True, )

    loss_criterion = loss.Loss()
    train(args, train_args, train_databatch, loss_criterion)

