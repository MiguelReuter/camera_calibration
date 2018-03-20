# encoding : utf-8

import numpy as np
import cv2

def load_points(filename):		
	with open(filename, 'r') as file:
		points = file.readlines()
	
	points = [point.split() for point in points]
	points =  np.matrix([[float(c) for c in p] for p in points]) 

	return points

def get_residual_error(pts2, pts3, m):
	n = pts2.shape[0]
	
	r = .0
	for i in range(n):
		p3 = np.matrix(np.append(np.array(pts3[i, :]), ([[1]]))).T
		p2 = np.matrix(np.append(np.array(pts2[i, :]), ([[1]]))).T

		p2_proj = m * p3
		p2_proj /= p2_proj[-1]
		
		dr = np.sqrt(((p2 - p2_proj).T*(p2 - p2_proj))[0, 0])
		r += dr
	r /= n

	return r


def compute_projection_matrix(pts_2d, pts_3d):
	n = pts_2d.shape[0]
	A = []

	for pt_i in range(n):
		pt_2 = pts_2d[pt_i, :].tolist()[0]
		pt_3 = pts_3d[pt_i, :].tolist()[0]
		A.append([*[pt_3[i] for i in range(3)], 1, *[0]*4, *[-pt_2[0] * pt_3[i] for i in range(3)], -pt_2[0]])
		A.append([*[0]*4, *[pt_3[i] for i in range(3)], 1, *[-pt_2[1] * pt_3[i] for i in range(3)], -pt_2[1]])
	A = np.array(A)
	
	[U, S, V] = np.linalg.svd(A)
	M = V[-1, :]
	M = M.reshape((3, 4))
	
	# get error for last point
	pt3 = np.matrix(np.append(np.array(pts_3d[-1, :]), ([[1]]))).T
	pt2_proj = M * pt3
	pt2_proj /= pt2_proj[-1]

	pt2 = np.matrix(np.append(np.array(pts_2d[-1, :]), ([[1]]))).T

	r = np.sqrt(((pt2 - pt2_proj).T*(pt2 - pt2_proj))[0, 0])

	return M, pt2_proj, r


def find_best_projection_matrix(pts2, pts3, k=(8, 12, 16), n=10):
	pts_nb = pts2.shape[0]
	
	average_residual_error_matrix = np.zeros((len(k), n))

	best_m = None
	best_r = 1E9	
	
	for i, ki in enumerate(k):
		for j in range(n):
			# choose ki pts for projection matrix computation
			rand_idx = list(range(pts_nb))
			np.random.shuffle(rand_idx)
			# indexes for projection matrix computaion and residual error computation
			comp_id = rand_idx[:ki]
			# choose 4 pts for residual error computation
			test_id = rand_idx[ki:ki+4]
		
			[m_ki_j, _, _] = compute_projection_matrix(pts2[comp_id,:], pts3[comp_id,:])
			r_ki_j = get_residual_error(pts2[test_id,], pts3[test_id,:], m_ki_j)
			
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
	A = []
	
	# construct A matrix for F computation
	for i in range(n):
		pt_ai = np.append(np.array(pts2_a[i, :]), ([[1]]))
		pt_bi = np.append(np.array(pts2_b[i, :]), ([[1]]))
		l = [pt_ai[k] * pt_bi[j] for j in range(3) for k in range(3)]
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

def compute_epipolar_lines(pts2_a, pts2_b, F):
	n = pts2_a.shape[0]

	picA_lines = np.zeros((n, 3))
	picB_lines = np.zeros((n, 3))

	for i in range(n):
		pt_ai = np.matrix(np.append(np.array(pts2_a[i, :]), ([[1]])))
		pt_bi = np.matrix(np.append(np.array(pts2_b[i, :]), ([[1]])))
		
		li_a = np.matrix(np.array(F * pt_bi.T)).T
		picA_lines[i, :] = li_a / li_a[0, -1]
		
		li_b = np.matrix(np.array(F.T * pt_ai.T)).T
		picB_lines[i, :] = li_b / li_b[0, -1]

	return picA_lines, picB_lines
	

def draw_lines(pts_a, pts_b, l_a, l_b):
	im_a = cv2.imread("input/pic_a.jpg")
	im_b = cv2.imread("input/pic_b.jpg")
	
	# image a
	x_min = 0
	x_max = im_b.shape[1]
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



if __name__ == "__main__":
	# M_norm_A	
	pts_2d = load_points("input/pts2d-norm-pic_a.txt")
	pts_3d = load_points("input/pts3d-norm.txt")
	
	[_, M_norm_A] = find_best_projection_matrix(pts_2d, pts_3d)
	compute_camera_center(M_norm_A)
	r = get_residual_error(pts_2d, pts_3d, M_norm_A)
	print("residual error : ", r)	
	
	
	# M_B
	#pts_2d = load_points("input/pts2d-pic_b.txt")
	#pts_3d = load_points("input/pts3d.txt")
	
	#[r_mat, m] = find_best_projection_matrix(pts_2d, pts_3d)
	#r = get_residual_error(pts_2d, pts_3d, m)
	#compute_camera_center(m)
	#print("residual error : ", r)

	# fundamental matrix
	pts_2d_a = load_points("input/pts2d-pic_a.txt")
	pts_2d_b = load_points("input/pts2d-pic_b.txt")
	F = compute_fundamental_matrix(pts_2d_a, pts_2d_b)

	[lines_a, lines_b] = compute_epipolar_lines(pts_2d_a, pts_2d_b, F)

	draw_lines(pts_2d_a, pts_2d_b, lines_a, lines_b)



















