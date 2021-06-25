
# from utils.networks import NestedUNet #import your network
import argparse
import torch
import torch.backends.cudnn as cudnn
import time
def compute_speed(model, input_size, device, iteration=100):
    torch.cuda.set_device(device)
    cudnn.benchmark = True
    model.eval()
    model = model.cuda()
    input = torch.randn(*input_size, device=device)
    for _ in range(50):
        model(input)
    print('=========Speed Testing=========')
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(iteration):
        model(input)
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start
    speed_time = elapsed_time / iteration * 1000
    fps = iteration / elapsed_time
    print('Elapsed Time: [%.2f s / %d iter]' % (elapsed_time, iteration))
    print('Speed Time: %.2f ms / iter   FPS: %.2f' % (speed_time, fps))
    return speed_time, fps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='NestedUNet')
    parser.add_argument('--intputchannel',type=int,default=1)
    parser.add_argument('--outputchannel', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--iter', type=int, default=100)
    parser.add_argument("--gpus", type=str, default="0", help="gpu ids (default: 0)")
    args = parser.parse_args()
    h, w =512,512
    model=eval(args.mode)(args.intputchannel,args.outputchannel)
    compute_speed(model, (args.batch_size,args.intputchannel, h, w), int(args.gpus), iteration=args.iter)
