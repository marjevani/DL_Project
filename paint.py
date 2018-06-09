from tkinter import *
import numpy as np
import scipy.signal
import mnist_object
from util import *

ROWS = 20
COLS = 20

class Paint(object):

    def __init__(self):
        ## build GUI ##
        self.root = Tk()

        self.center_btn = Button(self.root, text='center', command=self.centralize)
        self.center_btn.grid(row=0, column=0)

        self.pad_btn = Button(self.root, text='pad', command=self.pad)
        self.pad_btn.grid(row=0, column=1)

        self.send_btn = Button(self.root, text='send', command=self.send_eval)
        self.send_btn.grid(row=0, column=2)

        # Create eraser/painter radio buttons:
        erase_frame = Frame(self.root)
        self.eraser_on = BooleanVar()
        self.eraser_on.set(False)
        R1 = Radiobutton(erase_frame, text="painter", variable=self.eraser_on, value=False)
        R2 = Radiobutton(erase_frame, text="eraser", variable=self.eraser_on, value=True)
        erase_frame.grid(row=0, column=4)
        R1.pack(side="left")
        R2.pack(side="right")

        # Create a grid of None to store the references to the tiles
        self.tiles = [[None for _ in range(COLS)] for _ in range(ROWS)]

        self.canvas = Canvas(self.root, bg='white', width=600, height=600)
        self.canvas.grid(row=1, columnspan=5)
        self.canvas.bind("<B1-Motion>", self.callback)

        self.root.mainloop()

    def callback(self,event):
        # Get rectangle diameters
        col_width = int(self.canvas.winfo_width() / COLS)
        row_height = int(self.canvas.winfo_height() / ROWS)
        # Calculate column and row number
        col = min(event.x//col_width, COLS-1)
        row = min(event.y//row_height, ROWS-1)
        # paint or erase rectangle
        if not self.eraser_on.get():
            if not self.tiles[row][col]:
                # If the tile is not filled, create a rectangle
                colorval = "#%02x%02x%02x" % (0, 0, 0)
                self.tiles[row][col] =self.canvas.create_rectangle(col * col_width, row * row_height, (col + 1) * col_width, (row + 1) * row_height, fill=colorval, outline=colorval)
        else:
            self.canvas.delete(self.tiles[row][col])
            self.tiles[row][col] = None

    def centralize(self):
        x_pad =[]
        for pad in range(-4,5):
            x_cent = 0
            for row in range(20):
                for col in range(20):
                    x_cent = x_cent + ((10 - col + pad) if not self.tiles[row][col] is None else 0)
            x_pad.append((pad,x_cent))

        y_pad = []
        for pad in range(-4, 5):
            y_cent = 0
            for row in range(20):
                for col in range(20):
                    y_cent = y_cent + ((10 - row + pad) if not self.tiles[row][col] is None else 0)
            y_pad.append((pad, y_cent))

        y_pad.sort(key=lambda x : np.abs(x[1]))
        x_pad.sort(key=lambda x: np.abs(x[1]))

        x_cent,y_cent = x_pad[0][0],y_pad[0][0]

        debug_print(x_cent,y_cent)

        lines_top = [[0] * 20]*  (4-y_cent )
        lines_bottom = [[0] * 20] * (4+y_cent )

        lines_left = [[0] *(4 - x_cent)] * 28
        lines_rigth = [[0] * (4 + x_cent)] * 28

        pic = []
        for row in self.tiles:
            tmp = []
            pic.append(tmp)
            for col in row:
                print_val = "{0:^3}".format(0 if col == None else 255)
                debug_print(print_val,end= " ")
                tmp.append(0 if col == None else 255)
            debug_print()


        col_width = int(self.canvas.winfo_width() / 28)
        row_height = int(self.canvas.winfo_height() / 28)
        for row in range(20):
            for col in range(20):
                self.canvas.delete(self.tiles[row][col])
                self.tiles[row][col] = None

        pic = lines_top + pic + lines_bottom
        debug_print(np.array(lines_left).shape, np.array(pic).shape, np.array(lines_rigth).shape)
        pic = np.concatenate((lines_left, pic, lines_rigth), axis=1)

        centerd = [[None]*28]*28
        for row in range(28):
            for col in range(28):
                num = int(255- pic[row][col])
                colorval = "#%02x%02x%02x" % (num,num,num)
                centerd[row][col] = self.canvas.create_rectangle(col * col_width, row * row_height, (col + 1) * col_width,
                                                                 (row + 1) * row_height, fill=colorval, outline=colorval)
        self.pic = pic


    def pad(self):
        pic = self.pic
        filter = [
            [0.01, 0.01, 0.01],
            [0.01, 1, 0.01],
            [0.01, 0.01, 0.01]
        ]
        debug_print(filter)
        res = scipy.signal.convolve2d(pic, filter, mode='same')
        self.final = res
        for row in res:
            for col in row:
                print_val = "{0:^7.4f}".format(col)
                debug_print(print_val, end=" ")
            debug_print()

        col_width = int(self.canvas.winfo_width() / 28)
        row_height = int(self.canvas.winfo_height() / 28)

        centerd = [[None] * 28] * 28
        for row in range(28):
            for col in range(28):
                num = 255 - int(255 if res[row][col] * res[row][col] > 255 else res[row][col] * res[row][col])
                colorval = "#%02x%02x%02x" % (num, num, num)
                centerd[row][col] = self.canvas.create_rectangle(col * col_width, row * row_height, (col + 1) * col_width,
                                                                 (row + 1) * row_height, fill=colorval, outline=colorval)

    def send_eval(self):
        mnist_object.eval(self.final)


if __name__ == '__main__':
    Paint()