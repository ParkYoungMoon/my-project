#!/usr/bin/env python3
'''
start factory
'''
import os
import threading
import sys
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep
import numpy as np
import openvino as ov
import cv2
from iotdemo import FactoryController, MotionDetector, ColorDetector
FORCE_STOP = False

def thread_cam1(q_):
    # Initialize motion detector and OpenVINO
    '''
    Thread1 start
    '''
    det = MotionDetector()
    det.load_preset('resources/motion.cfg', 'default')
    model_path = 'resources/openvino.xml'
    core = ov.Core()
    model = core.read_model(model_path)
    ppp = ov.preprocess.PrePostProcessor(model)
    cap = cv2.VideoCapture('resources/conveyor.mp4')
    flag = True

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break
        q_.put(("cam1", frame))
        detected = det.detect(frame)
        if detected is None:
            continue
        q_.put(('VIDEO: Cam1 detected', detected))
        input_tensor = np.expand_dims(detected, 0)

        if flag:
            _, _, _, _ = input_tensor.shape
            ppp.input().tensor()\
                .set_shape(input_tensor.shape)\
                .set_element_type(ov.Type.u8)\
                .set_layout(ov.Layout('NHWC'))
            ppp.input().preprocess().resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)
            ppp.input().model().set_layout(ov.Layout('NCHW'))
            ppp.output().tensor().set_element_type(ov.Type.f32)
            model = ppp.build()
            compiled_model = core.compile_model(model, "CPU")
            flag = False

        results = compiled_model.infer_new_request({0: input_tensor})
        predictions = next(iter(results.values()))
        probs = predictions.reshape(-1)

        print(f"{probs}")
        if probs[0] > 0.0:
            print("Not Good")
            q_.put(("PUSH", 1))
        else:
            print("Good")

    cap.release()
    q_.put(('DONE', None))
    sys.exit()

def thread_cam2(q_):
    # Initialize motion and color detectors
    '''
    Thread2 start
    '''
    det = MotionDetector()
    det.load_preset('resources/motion.cfg', 'default')

    color = ColorDetector()
    color.load_preset('resources/color.cfg', 'default')

    cap = cv2.VideoCapture('resources/conveyor.mp4')

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break
        q_.put(("cam2", frame))
        detected = det.detect(frame)
        if detected is None:
            continue
        q_.put(('VIDEO: Cam2 detected', detected))

        predict = color.detect(detected)
        name, ratio = predict[0]
        _, ratio2 = predict[1]

        n_ratio = ratio / (ratio + ratio2) * 100
        print(f"{name}: {n_ratio:.2f}%")
        if name == 'blue':
            q_.put(('PUSH', 2))

    cap.release()
    q_.put(('DONE', None))
    sys.exit()

def imshow(title, frame, pos=None):
    '''
    imshow
    '''
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)

def main():
    '''
    main start
    '''
    global FORCE_STOP
    parser = ArgumentParser(prog='python3 factory.py',
                            description="Factory tool")
    parser.add_argument("-d",
                        "--device",
                        default=None,
                        type=str,
                        help="Arduino port")
    args = parser.parse_args()
    q = Queue()
    t_1 = threading.Thread(target=thread_cam1, args=(q,))
    t_2 = threading.Thread(target=thread_cam2, args=(q,))
    t_1.start()
    t_2.start()
    with FactoryController(args.device) as ctrl:
        ctrl.system_start()
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break
            try:
                name, frame = q.get(timeout=1)
                if name == 'PUSH':
                    ctrl.push_actuator(frame)
                elif name:
                    imshow(name, frame)
                q.task_done()
                if name == 'DONE':
                    FORCE_STOP = True
            except Empty:
                pass
    t_1.join()
    t_2.join()
    cv2.destroyAllWindows()
    ctrl.system_stop()
    ctrl.close()
if __name__ == '__main__':
    try:
        main()
    except OSError:
        os._exit(0)  # Use the correct syntax for exiting the script
