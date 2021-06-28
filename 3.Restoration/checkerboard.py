import numpy as np

def checkerboard(s, m = 4, n = 4):
	bw_tile = np.vstack((np.zeros((s, s)), np.ones((s, s))))
	wb_tile = np.vstack((np.ones((s, s)), np.zeros((s, s))))
	bg_tile = np.vstack((np.zeros((s, s)), np.full((s, s), 0.7)))
	gb_tile = np.vstack((np.full((s, s), 0.7), np.zeros((s, s))))
	
	board = np.copy(bw_tile)
	for i in range (1, 2 * n):
		if (i < n):
			# White tiles
			if (i % 2):
				board = np.hstack((board, wb_tile))
			else:
				board = np.hstack((board, bw_tile))
		else:
			# Gray tiles
			if (i % 2):
				board = np.hstack((board, gb_tile))
			else:
				board = np.hstack((board, bg_tile))
	
	board = np.tile(board, [m, 1])	# Complete board
	
	return board