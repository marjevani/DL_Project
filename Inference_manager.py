import paint
import mnist_object
from util import *
import scipy.signal
import numpy as np

class Inference_manager(object):

    def __init__(self):
        paint.Paint(self)

    def pre_process(self, tile):
        self.orig_img = tile
        self.centralize()
        self.pad()
        return self.processed_img

    def centralize(self):
        x_pad = []
        for pad in range(-4, 5):
            x_cent = 0
            for row in range(20):
                for col in range(20):
                    x_cent = x_cent + ((10 - col + pad) if not self.orig_img[row][col] is None else 0)
            x_pad.append((pad, x_cent))

        y_pad = []
        for pad in range(-4, 5):
            y_cent = 0
            for row in range(20):
                for col in range(20):
                    y_cent = y_cent + ((10 - row + pad) if not self.orig_img[row][col] is None else 0)
            y_pad.append((pad, y_cent))

        y_pad.sort(key=lambda x: np.abs(x[1]))
        x_pad.sort(key=lambda x: np.abs(x[1]))

        x_cent, y_cent = x_pad[0][0], y_pad[0][0]

        debug_print(x_cent, y_cent)

        lines_top = [[0] * 20] * (4 - y_cent)
        lines_bottom = [[0] * 20] * (4 + y_cent)

        lines_left = [[0] * (4 - x_cent)] * 28
        lines_rigth = [[0] * (4 + x_cent)] * 28

        pic = []
        for row in self.orig_img:
            tmp = []
            pic.append(tmp)
            for col in row:
                print_val = "{0:^3}".format(0 if col == None else 255)
                debug_print(print_val, end=" ")
                tmp.append(0 if col == None else 255)
            debug_print()

        pic = lines_top + pic + lines_bottom
        debug_print(np.array(lines_left).shape, np.array(pic).shape, np.array(lines_rigth).shape)
        pic = np.concatenate((lines_left, pic, lines_rigth), axis=1)

        # self.painter.show_img(pic)
        self.processed_img = pic


    def pad(self):
        filter = [
            [0.01, 0.01, 0.01],
            [0.01, 1, 0.01],
            [0.01, 0.01, 0.01]
        ]
        debug_print(filter)
        res = scipy.signal.convolve2d(self.processed_img, filter, mode='same')
        self.processed_img = res
        for row in res:
            for col in row:
                print_val = "{0:^7.4f}".format(col)
                debug_print(print_val, end=" ")
            debug_print()

    def send_eval(self):
        mnist_object.eval(self.processed_img)


def main():
    Inference_manager()

if __name__ == '__main__':
    main()
