import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from stockfish import Stockfish
from mss import mss

stockfish = Stockfish(r'C:/Users/15193/OneDrive/PYTHON/ChessDetect/SF/stockfish-11-win/stockfish-11-win/Windows/stockfish_20011801_x64.exe')

# GET IMAGES

# imc = cv.imread('ChessDetect/test/test6.jpg')
# im = cv.imread('ChessDetect/test/test6.jpg', cv.IMREAD_GRAYSCALE)

bounding_box = {'top': 293, 'left': 64, 'width': 658, 'height': 658}
sct = mss()

im = sct.grab(bounding_box)
im = np.array(im)
imc = im    
im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

# GET TURN

turn = input("playing as (w/b): ")

# GET BOARD PARAMS

# board_crop = im[uppermost - bound_safety:lowermost + bound_safety, leftmost - bound_safety:rightmost + bound_safety]
square_sl = int(im.shape[0] / 8)

# GET PIECE LOCATIONS

dataset = 'ChessDetect/templates/128h/'

board_data = np.zeros((8, 8), dtype=np.int32)
key_arr = ['b', 'k', 'n', 'p', 'q', 'r', 'B', 'K', 'N', 'P', 'Q', 'R']

icount = 0

for item in os.listdir(dataset):

    piece = cv.imread(dataset + item, cv.IMREAD_GRAYSCALE)
    piece_res = cv.resize(piece, (int(square_sl), int(square_sl)))

    tm_result = cv.matchTemplate(im, piece_res, cv.TM_CCOEFF_NORMED)
    
    threshold = 0

    if icount < 6:
        threshold = 0.85
    else:
        threshold = 0.7

    locations = np.where(tm_result >= threshold)
    locations = list(zip(*locations[::-1]))

    sqs = []
    for piece_loc in locations:
        file = piece_loc[0] * 8 / (im.shape[0])
        file = round(file)

        rank = (piece_loc[1] * 8 / (im.shape[0]))
        rank = round(rank)

        pos = [file, rank]

        sqs.append(pos)
    
    sqs_clear = []

    for sq in sqs:
        if sq not in sqs_clear:
            sqs_clear.append(sq)

    for sq in sqs_clear:
        r = sq[1]
        f = sq[0]
        board_data[r][f] = ord(key_arr[icount])

    icount = icount + 1

print(board_data)

# ROTATE BOARD MATRIX IF BLACK'S TURN

def rev_columns():
    for i in range (8):
        j = 0
        k = 7
        while j < k:
            t = board_data[j][i]
            board_data[j][i] = board_data[k][i]
            board_data[k][i] = t
            j = j + 1
            k = k - 1

if turn == 'b':
    board_data = np.transpose(board_data)
    rev_columns()
    board_data = np.transpose(board_data)
    rev_columns()

# GET FEN CODE

fen = ""

for r in range (8):
    zcount = 0
    for f in range (8):
        if board_data[r][f] == 0:
            zcount = zcount + 1
            if f == 7:
                fen = fen + str(zcount)
        else:
            if zcount > 0:
                fen = fen + str(zcount)
                zcount = 0
            fen = fen + chr(board_data[r][f])
    if r != 7: fen = fen + "/"

fen = fen + " " + turn + " - - 0 1"

print(fen)

# GET BEST MOVE

# print(stockfish.is_fen_valid(fen))

if stockfish.is_fen_valid(fen):
    stockfish.set_fen_position(fen)
    best_move = stockfish.get_best_move()

start_f, start_r, end_f, end_r = best_move[0], best_move[1], best_move[2], best_move[3]

if turn == 'w':
    start_pt = [round((ord(start_f) - ord('a')) * im.shape[0] / 8) + int(square_sl)/2, round((8 - int(start_r)) * im.shape[0] / 8) + int(square_sl)/2]
    end_pt = [round((ord(end_f) - ord('a')) * im.shape[0] / 8) + int(square_sl)/2, round((8 - int(end_r)) * im.shape[0] / 8) + int(square_sl)/2]
else:
    start_pt = [round((ord('h') - ord(start_f)) * im.shape[0] / 8) + int(square_sl)/2, round((int(start_r) - 1) * im.shape[0] / 8) + int(square_sl)/2]
    end_pt = [round((ord('h') - ord(end_f)) * im.shape[0] / 8) + int(square_sl)/2, round((int(end_r) - 1) * im.shape[0] / 8) + int(square_sl)/2]    

start_pt = np.int32(start_pt)
end_pt = np.int32(end_pt)

# DISPLAY

imc = cv.line(imc, start_pt, end_pt, (255, 255, 0), 2)
imc = cv.circle(imc, end_pt, radius=8, color=(255, 255, 0), thickness=5)

# out = im

# for pt in masked_board_pts:
#     out = cv.circle(out, (int(pt[0]), int(pt[1])), radius=5, color=(0, 255, 0), thickness=3)

plt.imshow(cv.cvtColor(imc, cv.COLOR_BGR2RGB))
plt.show()

# MUST SPECIFY WHICH COLOUR IS BEING PLAYED AS AND WHO'S TURN IT IS

##### TO DO #####
# create templates based on initial set up
# live play