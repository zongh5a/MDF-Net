import os, argparse
import conf

def moves(args):
    scans = None
    if args.dataset == "dtu":
        scans = conf.dtu_scans

    elif args.dataset == "tanks":
        scans = conf.tanks_scans

    else:
        print("exit: please use dtu or tanks, cuttent is", args.dataset)
        exit()

    os.makedirs(args.collect_folder, exist_ok=True)
    for scan in scans:
        # Move ply to dtu eval folder and rename
        scan_folder = os.path.join(conf.eval_folder, scan, "fuse")
        consis_folders = [f for f in os.listdir(scan_folder) if f.startswith('consistencyCheck-')]
        consis_folders.sort()
        consis_folder = consis_folders[-1]
        source_ply = os.path.join(scan_folder, consis_folder, 'final3d_model.ply')
        if args.dataset == "dtu":
            scan_idx = int(scan[4:])
            target_ply = os.path.join(args.collect_folder, 'ours{:03d}_l3.ply'.format(scan_idx))
        elif args.dataset == "tanks":
            target_ply = os.path.join(args.collect_folder, '{:03d}.ply'.format(scan))

        cmd = 'cp '+source_ply+' '+ target_ply

        print(cmd)
        os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DTU eval parameter setting')
    parser.add_argument("-t", '--dataset', default='dtu', type=str, help='dtu, tanks')
    parser.add_argument("-c", '--collect_folder', default=conf.col_path, type=str, help='')   

    args = parser.parse_args()

    moves(args)

