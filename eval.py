import config

import os, time, logging, argparse
import torch
from torch.utils.data import DataLoader

from tools.data_io import save_pfm,write_depth_img


def eval(args, eval_args, eval_databatch):
    # creat model
    model = config.model

    # load breakpoint
    if args.pre_model is not None:
        checkpoint = torch.load(args.pre_model) #map_location=torch.device("cpu")   ,map_location=DEVICE
        model.load_state_dict(checkpoint["model"])  # strict=True

    # load to device
    model.to(eval_args.DEVICE)

    #eval
    model.eval()
    with torch.no_grad():
        for iteration, data in enumerate(eval_databatch):
            torch.cuda.empty_cache()
            data_batch = {k: v.to(eval_args.DEVICE) for k, v in data.items() if isinstance(v, torch.Tensor)}

            start_time = time.time()
            outputs = model(data_batch["imgs"], data_batch["extrinsics"], data_batch["intrinsics"], data_batch["depth_range"])
            logging.info("batch:"+str(iteration + 1)+ "/"+ str(len(eval_databatch))+" time = {:.3f}".format(time.time() - start_time)+
                    " memory:"+str(torch.cuda.max_memory_allocated()/(1024**2))+"MB")
                    
            del data_batch
            # save depth map,confidence map, depth img
            for filename, depth, photometric_confidence in \
                    zip(data["filename"], outputs["depth"], outputs["confidence"]):  #(B,H,W)
                depth_filename = os.path.join(eval_args.output_path, filename.format('depth_est', '.pfm'))
                depthimg_filename = os.path.join(eval_args.output_path, filename.format('depth_est', '.png'))
                confidence_filename = os.path.join(eval_args.output_path, filename.format('confidence', '.pfm'))

                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)

                # save depth maps
                save_pfm(depth_filename, depth.cpu())
                write_depth_img(depthimg_filename, depth.cpu().numpy())
                # Save prob maps
                save_pfm(confidence_filename, photometric_confidence.cpu())
                print("save depth file in:", depth_filename)

            # if iteration>5:
            #     break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tanks eval parameter setting')
    parser.add_argument('-p', '--pre_model', default=None, type=str, help='Pre training model')
    parser.add_argument('-d', '--dataset', default='dtu', type=str, choices=['dtu', 'tanks'], help='Set dataset')
    parser.add_argument('-s', '--set', default='intermediate', type=str,
                        choices=['intermediate', 'advanced'], help='The set of tanks dataset')

    args = parser.parse_args()
    logging.info(args)

    eval_args, eval_dataset = None, None
    if args.dataset == "dtu":
        load_args = config.LoadDTU()
        eval_args = config.EvalDTU()
        from load.dtueval import LoadDataset
        eval_dataset = LoadDataset(datasetpath=load_args.eval_root,
                                   pairpath=load_args.eval_pair,
                                   scencelist=load_args.eval_label,
                                   nviews=eval_args.nviews)
    elif args.dataset == "tanks":
        load_args = config.LoadTanks(tanks_set = args.set)
        eval_args = config.EvalTanks()
        from load.tankseval import LoadDataset
        eval_dataset = LoadDataset(datasetpath=load_args.eval_root,
                                   scenelist=load_args.scenelist,
                                   nviews=eval_args.nviews)
    else:
        print("Error dataset")
        exit()

    # load dataset
    eval_databatch = DataLoader(eval_dataset,
                                batch_size=eval_args.batch_size,
                                num_workers=eval_args.nworks,
                                shuffle=False, pin_memory=True, drop_last=False)

    eval(args, eval_args, eval_databatch)

