
import numpy as np
from picamera2 import Picamera2, Preview
import cv2
import time
from PIL import Image
import sys
import tkinter as tk
from tkinter import ttk
from datetime import datetime
import threading
import queue

# Set path
import pathlib
sys.path.append('/'.join(str(pathlib.Path(__file__).parent.resolve()).split('/')[:-1]))

from src.classes.imagereader import ImageReader
from src.classes.zernikesolver import ZernikeSolver

def preview_window(picam2):
    camera_config = picam2.create_preview_configuration()
    picam2.configure(camera_config)
    picam2.start_preview(Preview.QTGL)
    time.sleep(10)
    picam2.stop_preview(Preview.QTGL)

def show_preview(preview_queue, stop_event):
    cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Preview", 600, 400)
    
    while not stop_event.is_set():
        gray_8bit = preview_queue.get()
        cv2.imshow("Preview", gray_8bit)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

def read_cam(txt, c, update_queue):
    picam2 = Picamera2()
    
    max_val = 1 << 16
    exposure = 1000
    gain = 1.0
    date = datetime.today().strftime('%m_%d_%Y')

    max_iter = 100
    pct = .10  # percent increase/decrease of exposure time and gain
    ct_list = ["ExposureTime", "AnalogueGain"]
    counter = 0

    config = picam2.create_still_configuration(raw={"size": picam2.sensor_resolution})
    picam2.configure(config)
    picam2.set_controls({"ExposureTime": exposure, "AnalogueGain": gain})
    picam2.start()
    time.sleep(2)
    #picam2.capture_file("test.jpg")

    gray = None
    while True:

        if (counter > max_iter):
            break

        # all the camera commands + processing, need to time it
        raw = picam2.capture_array("raw")  # i could get fancy and do multiple captures, doubt it's necessary
        raw = raw[:3040, :6084]
        raw16 = raw.astype(np.uint16)  # raw array is 8 bit, convert to 16 bit. don't know why it's necessary but it is!...

        im = np.zeros((3040, 4056), dtype=np.uint16)  # this doesn't have to be in loop

        for byte in range(2):
            im[:, byte::2] = ( (raw16[:, byte::3] << 4) | ((raw16[:, 2::3] >> (byte * 4)) & 0b1111) )

        im16 = im * 16
        im = im16

        # rgb = cv2.cvtColor(im, cv2.COLOR_BAYER_BG2RGB)
        gray = cv2.cvtColor(im, cv2.COLOR_BAYER_BG2GRAY)
        # Add these lines to convert the gray image to 8-bit
        gray_8bit = gray // 256
        gray_8bit = gray_8bit.astype(np.uint8)

        # Start the preview thread and pass the gray_8bit image using the preview_queue
        if not 'preview_thread' in locals():
            stop_event = threading.Event()
            preview_queue = queue.Queue()
            preview_thread = threading.Thread(target=show_preview, args=(preview_queue, stop_event))
            preview_thread.start()
        else:
            preview_queue.put(gray_8bit)

        arrmax = np.amax(gray)
        if c[3] is None or (arrmax >  (0.8 * max_val)) or (arrmax < (0.4 * max_val)):
            
            update_queue.put(f"Max Saturation: {arrmax/max_val*100:.2f}%\nCalibrating...")
        else:
            update_queue.put(f"Max Saturation: {arrmax/max_val*100:.2f}%\nDefocus: {c[3]:.5f} μm")
        # print(f'arrmax = {arrmax}')
        # print(f"iter = {counter}")

        if (arrmax > (0.8 * max_val)):  # scene is too BRIGHT
            counter += 1
            exposure = int(np.floor((1 - pct) * exposure))
            gain = (1 - pct) * gain
            picam2.set_controls({"ExposureTime": exposure, "AnalogueGain": gain})
            metadata = picam2.capture_metadata()
            controls = {c: metadata[c] for c in ct_list}
            print(controls)
            

        elif (arrmax < (0.4 * max_val)):  # scene is too DARK
            counter += 1
            exposure = int(np.floor((1 + pct) * exposure))
            gain = (1 + pct) * gain
            picam2.set_controls({"ExposureTime": exposure, "AnalogueGain": gain})
            metadata = picam2.capture_metadata()
            controls = {c: metadata[c] for c in ct_list}
            print(controls)


        else:  # it is within range and can exit
            # exit = 1  # redundant
             # Perform wavefront reconstruction
            #print("size: ", len(gray_8bit), "type: ", type(gray_8bit))
            reader = ImageReader(imm_arr=gray_8bit, previews = False)
            grid = reader.grid
            coeffs = ZernikeSolver(grid).solve()
            # By assigning the values to the c array, we can acess it in the start_cam
            # function. We would also be able to add it to the UI
            for i in range(len(c)):
                c[i] = float(coeffs[i])
            update_queue.put(f"Max Saturation: {arrmax/max_val*100:.2f}%\nDefocus: {c[3]:.5f} μm")
            #print(c[3])
    stop_event.set()
    preview_thread.join()

   
def update_txt(txt, update_queue,root): #pass root as arg

    try:
        message = update_queue.get_nowait()
        txt.set(message)
    except queue.Empty:
        pass
    root.after(100, update_txt, txt, update_queue,root)
    
def start_cam():
    # Hold the coefficients
    c = [None] * 15

    root = tk.Tk()
    root.rowconfigure(0, minsize=200, weight=1)
    root.columnconfigure(0, minsize=400, weight=1)
    txt = tk.StringVar()
    lbl = tk.Label(root, textvariable=txt).grid(row=0, column=0)
    update_queue = queue.Queue() 
    
    task = threading.Thread(target=read_cam, daemon=True, args=(txt, c, update_queue)) #create and pass through update queue
    task.start()

    root.after(100, update_txt, txt, update_queue,root) #pass through root
    root.mainloop()


    sys.exit()

def not_sure():
    raw = picam2.capture_array("raw")  # i could get fancy and do multiple captures, doubt it's necessary
    raw = raw[:3040, :6084]
    raw16 = raw.astype(np.uint16)  # raw array is 8 bit, convert to 16 bit. don't think it's necessary...

    im = np.zeros((3040, 4056), dtype=np.uint16)  # this doesn't have to be in loop

    for byte in range(2):
        im[:, byte::2] = ( (raw16[:, byte::3] << 4) | ((raw16[:, 2::3] >> (byte * 4)) & 0b1111) )

    im16 = im * 16
    im = im16

    rgb = cv2.cvtColor(im, cv2.COLOR_BAYER_BG2RGB)

    gray = cv2.cvtColor(im, cv2.COLOR_BAYER_BG2GRAY)
