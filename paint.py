from tkinter import *
import time


ROWS = 20
COLS = 20

class Paint(object):

    def __init__(self, im):
        self.manager = im

        ## build GUI ##
        self.root = Tk()

        # self.center_btn = Button(self.root, text='center', command=self.centralize)
        # self.center_btn.grid(row=0, column=0)
        #
        # self.pad_btn = Button(self.root, text='pad', command=self.pad)
        # self.pad_btn.grid(row=0, column=1)

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


    def send_eval(self):
        processed = self.manager.pre_process(self.tiles)
        self.show_img(processed*processed)
        self.manager.send_eval()

    def show_img(self, img):
        # clear canvas
        self.canvas.delete("all");
        # paint img on canvas
        col_width = int(self.canvas.winfo_width() / 28)
        row_height = int(self.canvas.winfo_height() / 28)

        centerd = [[None] * 28] * 28
        for row in range(28):
            for col in range(28):
                num = int(255 - min(img[row][col], 255))
                colorval = "#%02x%02x%02x" % (num, num, num)
                centerd[row][col] = self.canvas.create_rectangle(col * col_width, row * row_height,
                                                                 (col + 1) * col_width,
                                                                 (row + 1) * row_height, fill=colorval,
                                                                 outline=colorval)




# if __name__ == '__main__':
#     p = Paint()
