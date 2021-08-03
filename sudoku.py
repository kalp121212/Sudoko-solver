import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from imutils import contours
from skimage.segmentation import clear_border
import inspect, sys, re, operator
from model import Trainer
from solver import Solver

class Detector:
    def __init__(self):
        p = re.compile("stage_(?P<idx>[0-9]+)_(?P<name>[a-zA-Z0-9_]+)")

        self.stages = list(sorted(
        map(
            lambda x: (p.fullmatch(x[0]).groupdict()['idx'], p.fullmatch(x[0]).groupdict()['name'], x[1]),
            filter(
                lambda x: inspect.ismethod(x[1]) and p.fullmatch(x[0]),
                inspect.getmembers(self))),
        key=lambda x: x[0]))

        # For storing the recognized digits
        self.digits = [ [0 for i in range(9)] for j in range(9) ]
        self.allowed = [ [False for i in range(9)] for j in range(9) ]

    # Takes as input 9x9 array of numpy images
    # Combines them into 1 image and returns
    # All 9x9 images need to be of same shape
    def makePreview(images):
        assert isinstance(images, list)
        assert len(images) > 0
        assert isinstance(images[0], list)
        assert len(images[0]) > 0
        assert isinstance(images[0], list)

        rows = len(images)
        cols = len(images[0])

        cellShape = images[0][0].shape

        padding = 10
        shape = (rows * cellShape[0] + (rows + 1) * padding, cols * cellShape[1] + (cols + 1) * padding)

        result = np.full(shape, 255, np.uint8)

        for row in range(rows):
            for col in range(cols):
                pos = (row * (padding + cellShape[0]) + padding, col * (padding + cellShape[1]) + padding)

                result[pos[0]:pos[0] + cellShape[0], pos[1]:pos[1] + cellShape[1]] = images[row][col]

        return result

    # Takes as input 9x9 array of digits
    # Prints it out on the console in the form of sudoku
    # None instead of number means that its an empty cell
    def showSudoku(array):
        cnt = 0
        for row in array:
            if cnt % 3 == 0:
                print('+-------+-------+-------+')

            colcnt = 0
            for cell in row:
                if colcnt % 3 == 0:
                    print('| ', end='')
                print('. ' if cell == 0 else str(cell) + ' ', end='')
                colcnt += 1
            print('|')
            cnt += 1
        print('+-------+-------+-------+')

    # Runs the detector on the image at path, and returns the 9x9 solved digits
    # if show=True, then the stage results are shown on screen
    # Corrections is an array of the kind [(1,2,9), (3,3,4) ...] which implies
    # that the digit at (1,2) is corrected to 9
    # and the digit at (3,3) is corrected to 4
    def run(self, path='assets/sudokus/sudoku2.jpg', show = False, corrections = []):
        self.path = path
        self.original = cv2.imread(path)

        self.run_stages(show)
        self.setting_digits()
        result = self.solve(corrections)


        if show:
            self.showSolved()
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return result

    # Runs all the stages
    def run_stages(self, show):
        results = [('Original', self.original)]

        for idx, name, fun in self.stages:
            image = fun().copy()
            results.append((name, image))

        if show:
            for name, img in results:
                cv2.imshow(name, img)

    # Stages
    # Stage function name format: stage_[stage index]_[stage name]
    # Stages are executed increasing order of stage index
    # The function should return a numpy image, which is shown onto the screen
    # In case you have 81 images of 9x9 sudoku cells, you can use makePreview()
    # to create a single image out of those
    # You can pass data from one stage to another using class member variables
    def stage_1_preprocess(self):
        image = cv2.cvtColor(self.original.copy(), cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (3, 3), 6)
        image = cv2.adaptiveThreshold(image, 255,1,1,11,2)
        self.image1=image
        return image

    def stage_2_reshape(self):
        # find contours in the thresholded image and sort them by size in
        # descending order
        cnts = cv2.findContours(self.image1.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # initialize a contour that corresponds to the puzzle outline
        puzzleCnt = None
    # loop over the contours
        for c in cnts:
        # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we can
        # assume we have found the outline of the puzzle
            if len(approx) == 4:
                puzzleCnt = approx
                break
    # check to see if we are visualizing the outline of the detected
    # Sudoku puzzle
        # draw the contour of the puzzle on the image and then display
        # it to our screen for visualization/debugging purposes
        temp  = self.original.copy()
        cv2.drawContours(temp, [puzzleCnt], -1, (0, 255, 0), 3)
        gray=cv2.cvtColor(self.original,cv2.COLOR_BGR2GRAY)
        self.image2 = four_point_transform(gray,puzzleCnt.reshape(4,2))
        return self.image2

    def stage_3_extract_cells(self):
      #  self.image2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)
#        image = cv2.bitwise_not(cv2.adaptiveThreshold(self.image2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 1))
        if self.path=='assets/sudokus/sudoku1.jpg':
            self.image2=self.image2[6:,3:]
        grid=self.image2
        edge_h = np.shape(grid)[0]
        edge_w = np.shape(grid)[1]
        celledge_h = edge_h // 9
        celledge_w = np.shape(grid)[1] // 9

        ans=[]
        for i in range(9):
            temp=[]
            for j in range(9):
                startx=j*celledge_w
                starty=i*celledge_h
                endx=(j+1)*celledge_w
                endy=(i+1)*celledge_h
                cell = self.image2[starty:endy, startx:endx]
                cell = cv2.threshold(cell, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                cell = clear_border(cell)
                thresh=cell
          #      cell=cv2.resize(cell,(28,28))
          #      temp.append(cell)
                # find contours in the thresholded cell
                cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
    # if no contours were found than this is an empty cell
                if len(cnts) == 0:
                    temp.append(np.zeros((28,28)))
                    continue
        # otherwise, find the largest contour in the cell and create a
        # mask for the contour
                c = max(cnts, key=cv2.contourArea)
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                (h, w) = thresh.shape
                percentFilled = cv2.countNonZero(mask) / float(w * h)
    # if less than 3% of the mask is filled then we are looking at
    # noise and can safely ignore the contour
                if percentFilled < 0.03:
                    temp.append(np.zeros((28,28)))
                    continue
    # apply the mask to the thresholded cell
                digit = cv2.bitwise_and(thresh, thresh, mask=mask)
                digit =cv2.resize(digit,(28,28))
                digit = cv2.fastNlMeansDenoising(digit)
                self.allowed[i][j]=True
                temp.append(digit)
          #      digit = cv2.resize(digit, (0,0), fx=16, fy=16)
           #     cv2.imshow("Digit",digit)
            #    cv2.waitKey()
            ans.append(np.array(temp))
        ans=np.array(ans)
        cells=[[ans[i][j] for j in range(9)] for i in range(9)]
        self.cells=ans
        return Detector.makePreview(cells)

    def setting_digits(self):
        t=Trainer()
        t.load_model()
        for i in range(9):
            for j in range(9):
                if self.allowed[i][j]:
                    self.digits[i][j]=int(t.predict(self.cells[i][j]))
                #    print(self.digits[i][j])

    # Solve function
    # Returns solution
    def solve(self, corrections):
        # Only upto 3 corrections allowed
        assert len(corrections) <= 3

        # Apply the corrections
        for i,j,x in corrections:
            self.digits[i][j]=x
        # Solve the sudoku
        self.answers = [[ self.digits[j][i] for i in range(9) ] for j in range(9)]
        s = Solver(self.answers)
        if s.solve():
            self.answers = s.digits
            return s.digits

        return s.digits

    # Optional
    # Use this function to backproject the solved digits onto the original image
    # Save the image of "solved" sudoku into the 'assets/sudoku/' folder with
    # an appropriate name
    def showSolved(self):
        pass


if __name__ == '__main__':
    d = Detector()
    result = d.run('assets/sudokus/sudoku1.jpg', show=True)
    d.setting_digits()
    print('Recognized Sudoku:')
    Detector.showSudoku(d.digits)
    print('\n\nSolved Sudoku:')
    Detector.showSudoku(d.answers)
