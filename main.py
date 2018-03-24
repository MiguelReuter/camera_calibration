# encoding : utf-8

import numpy as np
import cv2


def load_points(filename):		
	with open(filename, 'r') as file:
		points = file.readlines()
	
	points = [point.split() for point in points]
	
	# add homogeneous coord
	points =  np.matrix([[float(c) for c in p] + [1.0] for p in points]) 

	return points

def get_residual_error(pts2, pts3, m):
	n = pts2.shape[0]
	
	r = .0
	for i in range(n):
		p3 = np.matrix(pts3[i, :]).T
		p2 = np.matrix(pts2[i, :]).T

		p2_proj = m * p3
		p2_proj /= p2_proj[-1]
		
		dr = np.sqrt(((p2 - p2_proj).T*(p2 - p2_proj))[0, 0])
		r += dr
	r /= n

	return r


def compute_projection_matrix(pts_2d, pts_3d):
	n = pts_2d.shape[0]
	
	# construct A for M computation
	A = []
	for pt_i in range(n):
		pt_2 = pts_2d[pt_i, :].tolist()[0]
		pt_3 = pts_3d[pt_i, :].tolist()[0]
		A.append([*[pt_3[i] for i in range(4)], *[0]*4, *[-pt_2[0] * pt_3[i] for i in range(3)], -pt_2[0]])
		A.append([*[0]*4, *[pt_3[i] for i in range(4)], *[-pt_2[1] * pt_3[i] for i in range(3)], -pt_2[1]])
	A = np.array(A)
	
	[U, S, V] = np.linalg.svd(A)
	M = V[-1, :]
	M = M.reshape((3, 4))

	return M


def find_best_mat(m_computation_func, error_func, pts_1, pts_2, k=(8, 10, 16), nb_iter=10):
	pts_nb = pts_1.shape[0]
	
	average_residual_error_matrix = np.zeros((len(k), nb_iter))

	best_m = None
	best_r = 1E9

	for i, ki in enumerate(k):
		for j in range(nb_iter):
			# choose ki pts for projection matrix computation
			rand_idx = list(range(pts_nb))
			np.random.shuffle(rand_idx)
			# indexes for projection matrix computaion and residual error computation
			comp_id = rand_idx[:ki]
			# choose 4 pts for residual error computation
			test_id = rand_idx[ki:ki+4]
		
			m_ki_j = m_computation_func(pts_1[comp_id,:], pts_2[comp_id,:])
			r_ki_j = error_func(pts_1[test_id,:], pts_2[test_id,:], m_ki_j)
			
			average_residual_error_matrix[i, j] = r_ki_j
			
			if r_ki_j < best_r:
				best_r = r_ki_j
				best_m = m_ki_j

	return average_residual_error_matrix, best_m
	

def compute_camera_center(M):
	[Q, m4] = np.split(M, [3], axis=1)
	C = - np.linalg.inv(Q) * np.matrix(m4)
	
	return C


def compute_fundamental_matrix(pts2_a, pts2_b):
	n = pts2_a.shape[0]
		
	# construct A matrix for F computation
	A = []
	for i in range(n):
		pt_ai = pts2_a[i, :]
		pt_bi = pts2_b[i, :]
		l = [pt_ai[0, k] * pt_bi[0, j] for j in range(3) for k in range(3)]
		A.append(l)	
	A = np.matrix(A)

	# solving for F
	[_, _, V] = np.linalg.svd(A)
	F = V[-1, :].reshape((3, 3))
	
	# linear computation for F
	[U, S, V] = np.linalg.svd(F)
	S[-1] = 0
	F = U * np.diag(S) * V

	return F

def get_fundamental_mat_error(pts_a, pts_b, f):
	d = np.diag((pts_b * f * pts_a.T))
	return np.absolute(d).sum()
	
def compute_epipolar_lines(pts2_a, pts2_b, F):
	lines_a = pts2_b * F.T
	lines_b = pts2_a * F
	
	# to normalized homogeneous coords
	for i in range(pts2_a.shape[0]):
		lines_a[i] /= lines_a[i,-1]
		lines_b[i] /= lines_b[i, -1]
	
	return lines_a, lines_b
	

def draw_lines(pts_a, pts_b, l_a, l_b):
	im_a = cv2.imread("input/pic_a.jpg")
	im_b = cv2.imread("input/pic_b.jpg")
	
	# image a
	x_min = 0
	x_max = im_a.shape[1]
	for i in range(pts_b.shape[0]):		
		y_min = int(pts_a[i, 1] + (x_min - pts_a[i, 0]) * (l_a[i, 0] / l_a[i, 1]))
		y_max = int(pts_a[i, 1] + (x_max - pts_a[i, 0]) * (l_a[i, 0] / l_a[i, 1]))
		cv2.line(im_a, (x_min, y_min), (x_max, y_max), (0,0,255), 1)

	# image b
	x_min = 0
	x_max = im_b.shape[1]
	for i in range(pts_a.shape[0]):		
		y_min = int(pts_b[i, 1] + (x_min - pts_b[i, 0]) * (l_b[i, 0] / l_b[i, 1]))
		y_max = int(pts_b[i, 1] + (x_max - pts_b[i, 0]) * (l_b[i, 0] / l_b[i, 1]))
		cv2.line(im_b, (x_min, y_min), (x_max, y_max), (0,0,255), 1)

	cv2.imshow("image a", im_a)
	cv2.imshow("image b", im_b)
	cv2.waitKey(0)


def normalize_points(pts):
	n = pts.shape[0]

	cu = pts[:, 0].mean()
	cv = pts[:, 1].mean()

	dc = np.matrix([[cu, cv, 0.0]] * n)
	s = 1 / np.absolute(pts - dc).max();

	T = np.matrix([[s, 0, -s*cu], [0, s, -s*cv], [0, 0, 1]])
	norm_pts = pts * T.T

	return norm_pts, T



if __name__ == "__main__":

	#########################################
	#										#
	#			PROJECTION MATRIX			#
	#										#
	#########################################

	# load 3D points (real scene) and 2D points (picture A)
	pts_2d = load_points("input/pts2d-norm-pic_a.txt")
	pts_3d = load_points("input/pts3d-norm.txt")

	# find best projection matrix M_norm_A
	[_, M_norm_A] = find_best_mat(compute_projection_matrix, get_residual_error, pts_2d, pts_3d)
	
	# compute camera center
	cam_center = compute_camera_center(M_norm_A)

	# get error
	err_M = get_residual_error(pts_2d, pts_3d, M_norm_A)
	print("projection matrix error : ", err_M)


	#########################################
	#										#
	#			FUNDAMENTAL MATRIX			#
	#										#
	#########################################

	# loads 2D points from picture A and B 
	pts_2d_a = load_points("input/pts2d-pic_a.txt")
	pts_2d_b = load_points("input/pts2d-pic_b.txt")

	# normalize 2D points between [-1, 1] * [-1, 1]
	pts_norm_a, Ta = normalize_points(pts_2d_a)
	pts_norm_b, Tb = normalize_points(pts_2d_b)
	
	# find best fundamental matrix 
	[_, F] = find_best_mat(compute_fundamental_matrix, get_fundamental_mat_error, pts_norm_a, pts_norm_b)
	# process F for real points (not normalized in [-1, -1]*[-1, -1])
	F = Tb.T * F * Ta

	# get error
	err_F = get_fundamental_mat_error(pts_2d_a, pts_2d_b, F)
	print("fundamental matrix error : ", err_F)

	# draw epipolar lines on pictures
	[lines_a, lines_b] = compute_epipolar_lines(pts_2d_a, pts_2d_b, F)
	draw_lines(pts_2d_a, pts_2d_b, lines_a, lines_b)

















