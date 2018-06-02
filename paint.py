from tkinter import *
from tkinter.colorchooser import askcolor
import numpy as np
import scipy.signal
import time
import mnist_object
ROWS = 20
COLS = 20

class Paint(object):
    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Tk()

        self.center_btn = Button(self.root, text='center', command=self.centralize)
        self.center_btn.grid(row=0, column=0)

        self.pad_btn = Button(self.root, text='pad', command=self.pad)
        self.pad_btn.grid(row=0, column=1)

        self.send_btn = Button(self.root, text='send', command=self.send_eval)
        self.send_btn.grid(row=0, column=2)

        ## Create eraser/painter radio buttons:
        erase_frame = Frame(self.root)
        self.eraser_on = BooleanVar()
        R1 = Radiobutton(erase_frame, text="painter", variable=self.eraser_on, value=False,
                         command=self.use_eraser)
        R2 = Radiobutton(erase_frame, text="eraser", variable=self.eraser_on, value=True,
                         command=self.use_eraser)
        erase_frame.grid(row=0, column=4)

        R1.pack(side="left")
        R2.pack(side="right")

        # Create a grid of None to store the references to the tiles
        self.tiles = [[None for _ in range(COLS)] for _ in range(ROWS)]

        self.c = Canvas(self.root, bg='white', width=600, height=600)
        self.c.grid(row=1, columnspan=5)

        self.setup()
        self.root.mainloop()

    def callback(self,event):
        # Get rectangle diameters
        col_width = int(self.c.winfo_width()/COLS)
        row_height = int(self.c.winfo_height()/ROWS)
        # Calculate column and row number
        col = event.x//col_width
        row = event.y//row_height
        # If the tile is not filled, create a rectangle
        if not self.eraser_on.get():
            if not self.tiles[row][col]:
                colorval = "#%02x%02x%02x" % (0, 0, 0)
                self.tiles[row][col] =self.c.create_rectangle(col*col_width, row*row_height, (col+1)*col_width, (row+1)*row_height, fill=colorval,outline=colorval)
        else:
            self.c.delete(self.tiles[row][col])
            self.tiles[row][col] = None

    def setup(self):
        self.old_x = None
        self.old_y = None
        #self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on.set(False)
        self.active_button = self.center_btn
        self.c.bind("<B1-Motion>", self.callback)
        # self.c.bind('<B1-Motion>', self.callback)
        # self.c.bind('<ButtonRelease-1>', self.reset)

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

        print(x_cent,y_cent)

        lines_top = [[0] * 20]*  (4-y_cent )
        lines_bottom = [[0] * 20] * (4+y_cent )

        lines_left = [[0] *(4 - x_cent)] * 28
        lines_rigth = [[0] * (4 + x_cent)] * 28

        pic = []
        for row in self.tiles:
            tmp = []
            pic.append(tmp)
            for col in row:
                print("{0:^3}".format(0 if col == None else 255),end= " ")
                tmp.append(0 if col == None else 255)
            print()


        col_width = int(self.c.winfo_width() / 28 )
        row_height = int(self.c.winfo_height() / 28 )
        for row in range(20):
            for col in range(20):
                self.c.delete(self.tiles[row][col])
                self.tiles[row][col] = None

        pic = lines_top + pic + lines_bottom
        print(np.array(lines_left).shape, np.array(pic).shape, np.array(lines_rigth).shape)
        pic = np.concatenate((lines_left, pic, lines_rigth), axis=1)

        centerd = [[None]*28]*28
        for row in range(28):
            for col in range(28):
                num = int(255- pic[row][col])
                colorval = "#%02x%02x%02x" % (num,num,num)
                centerd[row][col] = self.c.create_rectangle(col * col_width, row * row_height, (col + 1) * col_width,
                                                           (row + 1) * row_height, fill=colorval, outline=colorval)
        self.pic = pic


    def pad(self):
        pic = self.pic
        filter = [
            [0.01, 0.01, 0.01],
            [0.01, 1, 0.01],
            [0.01, 0.01, 0.01]
        ]
        print(filter)
        res = scipy.signal.convolve2d(pic, filter, mode='same')
        self.final = res
        for row in res:
            for col in row:
                print("{0:^7.4f}".format(col), end=" ")
            print()

        col_width = int(self.c.winfo_width() / 28)
        row_height = int(self.c.winfo_height() / 28)
        # for row in range(20):
        #     for col in range(20):
        #         self.c.delete(self.tiles[row][col])
        #         self.tiles[row][col] = None

        centerd = [[None] * 28] * 28
        for row in range(28):
            for col in range(28):
                num = 255 - int(255 if res[row][col] * res[row][col] > 255 else res[row][col] * res[row][col])
                colorval = "#%02x%02x%02x" % (num, num, num)
                centerd[row][col] = self.c.create_rectangle(col * col_width, row * row_height, (col + 1) * col_width,
                                                            (row + 1) * row_height, fill=colorval, outline=colorval)

    def send_eval(self):
        print("ok")
        mnist_object.eval(self.final)
        #self.eraser_on = False
        #self.color = askcolor(color=self.color)[1]
        #self.tiles = [[None for _ in range(COLS)] for _ in range(ROWS)]

    def use_eraser(self):
        pass


    def reset(self, event):
        self.old_x, self.old_y = None, None


if __name__ == '__main__':
    # filter = [
    #     [0.1 / 8, 0.1 / 8, 0.1 / 8],
    #     [0.1 / 8, 0.9, 0.1 / 8],
    #     [0.1 / 8, 0.1 / 8, 0.1 / 8]
    # ]
    # print(filter)
    #        res = sp.convolve2d(pic,filter,mode='same')

    # for row in filter:
    #     for col in row:
    #         print("{0:^3}".format(col), end=" ")
    #     print()

    Paint()

        #
# import tkinter as tk
#
# # Set number of rows and columns
# ROWS = 20
# COLS = 20
#
# # Create a grid of None to store the references to the tiles
# tiles = [[None for _ in range(COLS)] for _ in range(ROWS)]
#
# def callback(event):
#     # Get rectangle diameters
#     col_width = int(c.winfo_width()/COLS)
#     row_height = int(c.winfo_height()/ROWS)
#     # Calculate column and row number
#     col = event.x//col_width
#     row = event.y//row_height
#     # If the tile is not filled, create a rectangle
#     if not tiles[row][col]:
#         tiles[row][col] = c.create_rectangle(col*col_width, row*row_height, (col+1)*col_width, (row+1)*row_height, fill="black")
#     # If the tile is filled, delete the rectangle and clear the reference
#     # else:
#     #     c.delete(tiles[row][col])
#     #     tiles[row][col] = None
#
# # Create the window, a canvas and the mouse click event binding
# root = tk.Tk()
# c = tk.Canvas(root, width=500, height=500, borderwidth=5, background='white')
# c.pack()
# c.bind("<B1-Motion>", callback)
#
# root.mainloop()