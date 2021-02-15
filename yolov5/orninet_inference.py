import argparse
import datetime
import json
import requests
import sys

import torch.backends.cudnn as cudnn

from models.experimental import *
from utils.datasets import *
from utils.utils import *


def detect(opt, save_img=False):
    out, source, weights, view_img, save_txt, imgsz, post_results, post_url = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.post_results, opt.post_url
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Don't delete output folder
    # if os.path.exists(out):
    #     shutil.rmtree(out)  # delete output folder
    # os.makedirs(out)  # make new output folder

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            # Save output name as timestamp
            p = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                results = []
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if post_results: # Write to database
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        results.append({
                            'species': names[int(cls)],
                            'confidence': conf.item(), 
                            'bbox': xywh
                        })

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                if post_results:
                    payload = {
                        'categories': {'detections': results},
                        'name': p,
                        'timestamp': str(datetime.datetime.now())
                    }

                    try:
                        response = requests.post(post_url, json=json.dumps(payload))
                        if response.status_code != 200:
                            # print(f"Detection POST Error: \n{response.reason}")
                            s += f'\nDetection POST Error: \n{response.reason}\n'
                        else:
                            # print("Detection POST Successful")
                            s += '\nDetection POST Successful\n'
                    except Exception as ex:
                        s += f'\nDetection POST Error: \n{ex}\n'

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
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

    if save_txt or save_img:
        # print('Results saved to %s' % os.getcwd() + os.sep + out)
        print('Results saved to %s' % save_path)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    if post_results:
        print('Posted results to %s' % post_url)

    print('Done. (%.3fs)' % (time.time() - t0))


class InferenceOptions():
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
        self.conf_thres = 0.4
        # IOU threshold for NMS
        self.iou_thres = 0.5
        # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.device = ''
        # display results
        self.view_img = False
        # save results to *.txt
        self.save_txt = False
        # filter by class
        self.classes = 0
        # class-agnostic NMS
        self.agnostic_nms = False
        # augmented inference
        self.augment = False
        # update all models
        self.update = False
        # database path
        self.post_url = ''
        # insert image name and detection data into database
        self.post_results = False


if __name__ == '__main__':
    options = InferenceOptions()
    options.device = '0'
    options.weights = "./weights/yolov5s.pt"
    options.source = "./inference/images/birds.jpg"
    options.output = "./inference/output" # "/Users/dillon.donohue/source/orninet-app/images"
    options.save_txt = True
    options.post_results = True
    options.post_url = 'http://localhost:5000/api/post-detection'

    with torch.no_grad():
        if options.update:  # update all models (to fix SourceChangeWarning)
            for options.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
                detect(options)
                create_pretrained(options.weights, options.weights)
        else:
            detect(options)
