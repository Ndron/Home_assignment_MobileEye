import argparse
import cv2
import torch
import json
import numpy as np
from ultralytics import YOLO
import time
import queue
import threading
import jsonlines
#multiprocessing
from multiprocessing import Process
from threading import Thread
from queue import Queue

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StreamLoader:
    def __init__(self, stream ,model  ,queueSize = 100 ):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(stream)
        self.model = model
        assert self.stream.isOpened(), 'Cannot capture source'
        self.stopped = False
        self.Q = Queue(maxsize=queueSize)
    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self
    def update(self):
        while True:
            # ensure the queue has room in it
            if not self.Q.full():
                grabbed, frame = self.stream.read()
                    # if the `grabbed` boolean is `False`, then we have stop thread
                if not grabbed:
                    self.stop()
                    return
                #Model inference
                detections = self.model.predict(frame, classes=[0])
                # Get the bounding boxes  of the detected objects
                boxes_res = detections[0].boxes.xywh.cpu().tolist()
                result = {
                    'time': time.time(),
                    'bboxes_xywh': boxes_res
                }
                self.Q.put((result))
            else:
                with self.Q.mutex:
                    self.Q.queue.clear()

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def read(self):
        # return next frame in the queue
        return self.Q.get()

class DataWriter:
    def __init__(self, queueSize=1024):

        self.stopped = False
        self.final_result = []
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)

    def start(self):
        # start a thread to write info to json file
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                return
            # otherwise, ensure the queue is not empty
            if not self.Q.empty():
                (result) = self.Q.get()

                with jsonlines.open("result.jsonl", "a") as writer:  # for writing
                    writer.write(result)
            else:
                time.sleep(0.1)

    def save(self, result):
        # save next frame in the queue
        self.Q.put((result))

    def running(self):
        # indicate that the thread is still running
        time.sleep(0.2)
        return not self.Q.empty()

    def stop(self):
        # indicate that the t    def results(self):
        #         # return final result
        #         return self.final_resulthread should be stopped
        self.stopped = True


if __name__ == '__main__':
    # Define the parser
    parser = argparse.ArgumentParser()

    # Add the arguments
    parser.add_argument('-s', '--source', type=str, help='The input source.')
    # Parse the arguments
    args = parser.parse_args()
    # load model
    model_n = YOLO('yolov8n.pt')
    model_n.to(device)
    #start threads
    model_loader = StreamLoader(args.source , model_n).start()
    writer = DataWriter().start()
    while True:
        ((result)) = model_loader.read()
        print(result)
        writer.save((result))
        #pass
        if model_loader.stopped:
            break

    model_loader.stop()
    if writer.Q.qsize() == 0:

        writer.stop()
    print('The End')