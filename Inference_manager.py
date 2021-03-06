import math

from paint import Paint
import threading
import mnist_object
from util import *
import scipy.signal
import numpy as np
from tkinter import messagebox

### init Paint and provide function for his implementation.
### NOTICE - evaluate calculate in different thread to prevent GUI freezing

class Inference_manager(object):

    def __init__(self):
        Paint(self)

    def init_pic(self):
        self.processed_img = []
        debug_print("init_picture:")
        for row in self.orig_tiles:
            tmp = []
            self.processed_img.append(tmp)
            for col in row:
                tmp.append(0.0 if col == None else 1.0)
                print_val = "{0:^3}".format(0 if col == None else 1.0)
                debug_print(print_val, end=" ")
            debug_print()


    def pre_process(self, tile):
        self.orig_tiles = tile

        self.init_pic()
        self.centralize()
        self.smooth()

        if DEBUG:
            # print the final self.processed_img for evaluate:
            debug_print("img to evaluate:")
            for row in self.processed_img:
                for col in row:
                    print_val = "{0:^7.4f}".format(col)
                    debug_print(print_val, end=" ")
                debug_print()

        return self.prepare_to_paint()

    def prepare_to_paint(self):
        pic = []
        for row in self.processed_img:
            tmp = []
            pic.append(tmp)
            for col in row:
                pixel = int((1-col)*255)
                tmp.append(pixel)
        return pic


    def centralize(self):
        # centralize & pad
        x_pad = []
        for pad in range(-4, 5):
            x_cent = 0
            for row in range(20):
                for col in range(20):
                    x_cent = x_cent + ((10 - col + pad) if self.processed_img[row][col] else 0)
            x_pad.append((pad, x_cent))

        y_pad = []
        for pad in range(-4, 5):
            y_cent = 0
            for row in range(20):
                for col in range(20):
                    y_cent = y_cent + ((10 - row + pad) if self.processed_img[row][col] else 0)
            y_pad.append((pad, y_cent))

        y_pad.sort(key=lambda x: np.abs(x[1]))
        x_pad.sort(key=lambda x: np.abs(x[1]))

        x_cent, y_cent = x_pad[0][0], y_pad[0][0]

        debug_print("x and y centered:")
        debug_print(x_cent, y_cent)

        lines_top = [[0] * 20] * (4 - y_cent)
        lines_bottom = [[0] * 20] * (4 + y_cent)
        lines_left = [[0] * (4 - x_cent)] * 28
        lines_rigth = [[0] * (4 + x_cent)] * 28

        # Y axis center
        self.processed_img = lines_top + self.processed_img  + lines_bottom
        # X axis center
        self.processed_img = np.concatenate((lines_left, self.processed_img , lines_rigth), axis=1)
        debug_print("img after centralize:")
        debug_print(np.array(lines_left).shape, np.array(self.processed_img).shape, np.array(lines_rigth).shape)


    def smooth(self):
        eadge = 0.04
        filter = [
            [eadge, eadge,    eadge],
            [eadge, 1-8*eadge, eadge],
            [eadge, eadge,    eadge]
        ]
        debug_print("filter = ")
        debug_print(filter)

        self.processed_img = scipy.signal.convolve2d(self.processed_img, filter, mode='same')
        # self.processed_img = (self.processed_img, 1)
        # self.processed_img = (self.processed_img * self.processed_img)
        # non-linear smoothing
        for row in range(28):
            for col in range(28):
                # self.processed_img[row][col] = math.sqrt(self.processed_img[row][col])
                self.processed_img[row][col] = math.pow(self.processed_img[row][col], 0.7)


    def send_eval(self, painter):
        # create evaluate thread
        t = threading.Thread(target=self.send_eval_mt, args=[self.processed_img, painter])
        t.start()

    def send_eval_mt(self, processed_img, painter):
        digit = mnist_object.eval(processed_img)
        # show the result in different popup window
        painter.set_status("The digit '" + str(digit) + "' Evaluated!!!")
        messagebox.showinfo("you just paint the digit:", str(digit))