import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from stockfish import Stockfish
from mss import mss
import time

# STOCKFISH IMPORT

stockfish = Stockfish(r'C:/Users/15193/OneDrive/PYTHON/ChessDetect/SF/stockfish-11-win/stockfish-11-win/Windows/stockfish_20011801_x64.exe')
stockfish.set_position([])

# GET IMAGES

bounding_box = {'top': 293, 'left': 64, 'width': 658, 'height': 658}
sct = mss()

# GET TURN

playing = input("playing as (w/b): ")

# SEARCH PARAMS

board_data = np.zeros((8, 8))

dataset = 'ChessDetect/templates/128h/'
key_arr = ['b', 'k', 'n', 'p', 'q', 'r', 'B', 'K', 'N', 'P', 'Q', 'R']

b_start, b_end, w_start, w_end = [-1, -1], [-1, -1], [-1, -1], [-1, -1]
b_end_copy = [-1, -1]
w_end_copy = [-1, -1]
start_pt = [0, 0]
end_pt = [0, 0]

def rev_columns(b):
    for i in range (8):
        j = 0
        k = 7
        while j < k:
            t = b[j][i]
            b[j][i] = b[k][i]
            b[k][i] = t
            j = j + 1
            k = k - 1

# LIVE RUN

tcount = 0
static_count = 0
move_count = 0
transpose = False

while True:

    im = sct.grab(bounding_box)
    im = np.array(im)
    imc = im    
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

# GET BOARD PARAMS

    square_sl = int(im.shape[0] / 8)
    board_copy = board_data

# GET PIECE LOCATIONS

    icount = 0

    if tcount == 20:
        board_data = np.zeros((8, 8))
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

            # ROTATE BOARD MATRIX IF PLAYING AS BLACK

        if playing == 'b' and transpose == False:
            board_data = np.transpose(board_data)
            rev_columns(board_data)
            board_data = np.transpose(board_data)
            rev_columns(board_data)

            if move_count == 0:
                board_copy = np.transpose(board_copy)
                rev_columns(board_copy)
                board_copy = np.transpose(board_copy)
                rev_columns(board_copy)

            transpose = True

            # FIND MOVES PLAYED

        b_start_copy, b_end_copy, w_start_copy, w_end_copy = b_start, b_end, w_start, w_end
        
        if not(w_end[0] != -1 and b_end[0] == -1):
            b_start, b_end, w_start, w_end = [-1, -1], [-1, -1], [-1, -1], [-1, -1]
            if board_copy[0][2] == 107 and board_data[0][4] == 107:
                b_start = [0, 4]
                b_end = [0, 2]
            elif board_copy[0][6] == 107 and board_data[0][4] == 107:
                b_start = [0, 4]
                b_end = [0, 6]
            elif board_copy[7][2] == 75 and board_data[7][4] == 75:
                b_start = [7, 4]
                b_end = [7, 2]
            elif board_copy[7][6] == 75 and board_data[7][4] == 75:
                b_start = [7, 4]
                b_end = [7, 6] 
            else:               
                for r in range (8):
                    for f in range (8):
                        if board_copy[r][f] >= 98 and board_data[r][f] == 0:
                            b_start = [r, f]
                        elif board_copy[r][f] < 98 and board_data[r][f] >= 98:
                            b_end = [r, f]
                        elif board_copy[r][f] >= 66 and  board_copy[r][f] < 98 and board_data[r][f] == 0:
                            w_start = [r, f]
                        elif (board_copy[r][f] == 0 or board_copy[r][f] >= 98) and board_data[r][f] >= 66 and  board_data[r][f] < 98:
                            w_end = [r, f]
        else:
            if board_copy[0][2] == 107 and board_data[0][4] == 107:
                b_start = [0, 4]
                b_end = [0, 2]
            elif board_copy[0][6] == 107 and board_data[0][4] == 107:
                b_start = [0, 4]
                b_end = [0, 6]
            else:
                for r in range (8):
                    for f in range (8):
                        if board_copy[r][f] >= 98 and board_data[r][f] == 0:
                            b_start = [r, f]
                        elif board_copy[r][f] < 98 and board_data[r][f] >= 98:
                            b_end = [r, f]

        # print(b_start, b_end)
        # print(w_start, w_end)

        # print(board_data)

        # GET BEST MOVE

        # for black: get white move, send to stockfish, get black move

        if (b_end[0] == -1 and w_end[0] == -1) and (b_end_copy[0] != -1 and w_end_copy[0] != -1):
            if playing == 'w':
                w_move = str(chr(ord('a') + w_start_copy[1])) + str(8 - w_start_copy[0]) + str(chr(ord('a') + w_end_copy[1])) + str(8 - w_end_copy[0])
                b_move = str(chr(ord('a') + b_start_copy[1])) + str(8 - b_start_copy[0]) + str(chr(ord('a') + b_end_copy[1])) + str(8 - b_end_copy[0])
            # else:
            #     w_move = str(chr(ord('h') - w_start_copy[1])) + str(w_start_copy[0] + 1) + str(chr(ord('h') - w_end_copy[1])) + str(w_end_copy[0] + 1)
            #     b_move = str(chr(ord('h') - b_start_copy[1])) + str(b_start_copy[0] + 1) + str(chr(ord('h') - b_end_copy[1])) + str(b_end_copy[0] + 1)

            print(w_move, b_move)
            if w_move[0] >= 'a' and w_move[0] <= 'h':
                stockfish.make_moves_from_current_position([w_move, b_move])

                move_count = move_count + 1
                # transpose = False

        if (b_end[0] == -1 and w_end[0] == -1) and (b_end_copy[0] == -1 and w_end_copy[0] == -1):
            best_move = stockfish.get_best_move()

            start_f, start_r, end_f, end_r = best_move[0], best_move[1], best_move[2], best_move[3]

            start_pt = [round((ord(start_f) - ord('a')) * im.shape[0] / 8) + int(square_sl)/2, round((8 - int(start_r)) * im.shape[0] / 8) + int(square_sl)/2]
            end_pt = [round((ord(end_f) - ord('a')) * im.shape[0] / 8) + int(square_sl)/2, round((8 - int(end_r)) * im.shape[0] / 8) + int(square_sl)/2]

            start_pt = np.int32(start_pt)
            end_pt = np.int32(end_pt)

        tcount = 0

    tcount = tcount + 1



# # GET BEST MOVE

#     best_move = stockfish.get_best_move()

#     start_f, start_r, end_f, end_r = best_move[0], best_move[1], best_move[2], best_move[3]

#     if playing == 'w':
#         start_pt = [round((ord(start_f) - ord('a')) * im.shape[0] / 8) + int(square_sl)/2, round((8 - int(start_r)) * im.shape[0] / 8) + int(square_sl)/2]
#         end_pt = [round((ord(end_f) - ord('a')) * im.shape[0] / 8) + int(square_sl)/2, round((8 - int(end_r)) * im.shape[0] / 8) + int(square_sl)/2]
#     else:
#         start_pt = [round((ord('h') - ord(start_f)) * im.shape[0] / 8) + int(square_sl)/2, round((int(start_r) - 1) * im.shape[0] / 8) + int(square_sl)/2]
#         end_pt = [round((ord('h') - ord(end_f)) * im.shape[0] / 8) + int(square_sl)/2, round((int(end_r) - 1) * im.shape[0] / 8) + int(square_sl)/2]    

#     start_pt = np.int32(start_pt)
#     end_pt = np.int32(end_pt)

# # DISPLAY

#     imc = cv.line(imc, start_pt, end_pt, (255, 255, 0), 2)
#     imc = cv.circle(imc, end_pt, radius=8, color=(255, 255, 0), thickness=5)
    if (b_end[0] == -1 and w_end[0] == -1) and (b_end_copy[0] == -1 and w_end_copy[0] == -1):
            imc = cv.line(imc, start_pt, end_pt, (0, 0, 255), 2)
            imc = cv.circle(imc, end_pt, radius=8, color=(0, 0, 255), thickness=5)

    cv.imshow('ChessDetect', np.array(imc))

# EXIT

    key = cv.waitKey(1)
    if key == 27:
        break
