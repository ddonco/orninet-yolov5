import argparse
import time
import datetime
import json
import logging
import requests
from pathlib import Path
import threading

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadCSICam, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def save_image(path, image):
    try:
        cv2.imwrite(path, image)
        print(f'Image ({image.shape}) saved to: {path}')
    except Exception as ex:
        logging.error(f'Image save Error: \n{ex}')

def save_text():
    pass

# POST detection payload to app server
def post_request(post_url, payload):
    try:
        response = requests.post(post_url, json=json.dumps(payload))
        if response.status_code != 200:
            print(f'\nDetection POST Error: \n{response.reason}\n')
            logging.error(f'Detection POST Error: \n{response.reason}')
        else:
            print('\nDetection POST Successful\n')
    except Exception as ex:
        print(f'\nDetection POST Error: \n{ex}\n')
        logging.error(f'Detection POST Error: \n{ex}')

def detect(opt, save_img=False):
    source, weights, view_img, save_txt, imgsz, post_results, post_url, target = \
        opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.post_results, opt.post_url, opt.target
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))
    target_found = False

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadCSICam(source, img_size=imgsz, stride=stride)
    else:
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        if webcam:
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        else:
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        t_start = time.time()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                results = [] # POST results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if post_results and cls == target: # List of results to POST to app server
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        results.append({
                            'species': names[int(cls)],
                            'confidence': conf.item(), 
                            'bbox': xywh
                        })
                        target_found = True
                    
                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                if post_results and target_found: # Build POST payload dict
                    payload = {
                        'categories': {'detections': results},
                        'name': str(p),
                        'timestamp': str(datetime.datetime.now())
                    }

                    post_thread = threading.Thread(target=post_request, args=(post_url, payload,))
                    post_thread.start()

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img or target_found:
                target_found = False
                if dataset.mode == 'image':
                    save_thread = threading.Thread(target=save_image, args=(save_path, im0,))
                    save_thread.start()
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

        print(f'FPS: {1 / (time.time() - t_start):.2f}')

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


class DetectOptions():
    def __init__(self):
        # model.pt path(s)
        self.weights = 'yolov5s.pt'
        # inference source
        self.source = 'inference/images'
        # inference output folder
        self.output = 'inference/output'
        # inference size (pixels)
        self.img_size = 640
        # object confidence threshold
        self.conf_thres = 0.25
        # IOU threshold for NMS
        self.iou_thres = 0.45
        # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.device = ''
        # display results
        self.view_img = False
        # save results to *.txt
        self.save_txt = False
        # save confidences in save_txt labels
        self.save_conf = False
        # filter by class
        self.classes = None
        # class-agnostic NMS
        self.agnostic_nms = False
        # augmented inference
        self.augment = False
        # update all models
        self.update = False
        # save results to project/name
        self.project = 'runs/detect'
        # save results to project/name
        self.name = 'exp'
        # existing project/name ok, do not increment
        self.exist_ok = False
        # app server url
        self.post_url = ''
        # POST results to app server
        self.post_results = False
        # target class for posting images
        self.target = 0


if __name__ == '__main__':
    # check_requirements()

    options = DetectOptions()
    options.device = '0'
    options.weights = './weights/yolov5s.pt'
    options.source =  '0' # './inference/images/birds.jpg'
    options.output = './inference/output' # "/Users/dillon.donohue/source/orninet-app/images"
    options.img_size = 608 # (342, 608)
    options.target = 14
    options.conf_thres = 0.25
    options.classes = 14
    options.save_txt = False
    options.post_results = False
    options.post_url = 'http://localhost:5000/api/post-detection'
    options.log = '/home/dd/source/orninet-yolov3/yolov3/orninet.log'

    with torch.no_grad():
        if options.update:  # update all models (to fix SourceChangeWarning)
            for options.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(options)
                strip_optimizer(options.weights)
        else:
            detect(options)
