from mmdet3d.apis import LidarDet3DInferencer
import argparse

def run(args):
    inferencer = LidarDet3DInferencer(model=args.model, weights=args.weights)
    result = inferencer(dict(points=args.input))
    print(result['predictions'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='config.py')
    parser.add_argument('--weights', type=str, default='checkpoint.pth')
    parser.add_argument('--input', type=str, default='demo.bin')
    args = parser.parse_args()
    run(args)
