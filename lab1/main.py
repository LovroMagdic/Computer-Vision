import matplotlib.pyplot as plt
import scipy.misc as misc
import numpy as np

def recover_projective(Qs, Qd):
  (x1, y1), (x2, y2), (x3, y3), (x4, y4) = Qs[0], Qs[1], Qs[2], Qs[3]
  (x1_, y1_), (x2_, y2_), (x3_, y3_), (x4_, y4_) = Qd[0], Qd[1], Qd[2], Qd[3]
    
  matrix_A = np.array([[-x1,-y1,-1,0,0,0,x1*x1_,y1*x1_],
                      [0,0,0,-x1,-y1,-1,x1*y1_,y1*y1_],
                      [-x2,-y2,-1,0,0,0,x2*x2_,y2*x2_],
                      [0,0,0,-x2,-y2,-1,x2*y2_,y2*y2_],
                      [-x3,-y3,-1,0,0,0,x3*x3_,y3*x3_],
                      [0,0,0,-x3,-y3,-1,x3*y3_,y3*y3_],
                      [-x4,-y4,-1,0,0,0,x4*x4_,y4*x4_],
                      [0,0,0,-x4,-y4,-1,x4*y4_,y4*y4_]])
  
  matrix_b = np.array([[-x1_],
                      [-y1_],
                      [-x2_],
                      [-y2_],
                      [-x3_],
                      [-y3_],
                      [-x4_],
                      [-y4_]])
  
  U, S, Vt = np.linalg.svd(matrix_A, full_matrices=False)
  S_inv = np.diag(1/ S)
  x = Vt.T @ S_inv @ U.T
  x = x @ matrix_b

  return x

def recover_affine_diamond(Is_height, Is_width, Hd,Wd):

  (x1, y1), (x2, y2), (x3, y3) = (0,0), (Is_width-1,0), (0, Is_height-1) # originalne tocke

  (x1_, y1_), (x2_, y2_), (x3_, y3_) = (0, 0), (0, Hd*6), (Wd*6, 0) # destinacijske tocke

  matrix_A = np.array([[x1,y1, 0, 0, 1, 0],[0,0, x1, y1, 0, 1],[x2,y2, 0, 0, 1, 0],[0 , 0 ,x2, y2, 0, 1],[x3 , y3 ,0, 0, 1, 0],[0 , 0 ,x3, y3, 0, 1]]) # matrica koordinata
  
  matrix_b = np.array([[x1_],[y1_],[x2_],[y2_],[x3_],[y3_]]) # matrica destinacijskih koordinata

  x = np.linalg.solve(matrix_A, matrix_b)
  A = [[x[0][0], x[1][0]], [x[2][0], x[3][0]]]
  b = [[x[4][0]],[x[5][0]]]
  return A,b

def affine_bilin(Is, A,b, Hd, Wd):

    if len(Is.shape) == 2:
      row, col = Is.shape
      transformed_image = np.zeros((Hd, Wd), dtype=Is.dtype)
    else:
       row, col, ch = Is.shape
       transformed_image = np.zeros_like(Is)

    inverse = np.linalg.inv(A) + b
    for i in range(Hd):
        for j in range(Wd):

            original = np.dot(inverse, [j / Wd * col, i / Hd * row])
            x, y = original[0], original[1]

            if 0 <= x < col-1 and 0 <= y < row-1:
              x1, y1 = int(np.floor(x)), int(np.floor(y))
              x2, y2 = x1+1, y1+1
              dx, dy = x-x1, y-y1
              value = (Is[y1, x1]*(1 - dx)*(1 - dy) +Is[y1, x2] * dx * (1 - dy) +Is[y2, x1] * (1 - dx) * dy + Is[y2, x2] * dx * dy)
              transformed_image[i, j] = value
    return transformed_image[50:200,50:200]

def affine_nn(Is, A, b, Hd, Wd):

    if len(Is.shape) == 2:
      row, col = Is.shape
      transformed_image = np.zeros((Hd, Wd), dtype=Is.dtype)
    else:
       row, col, ch = Is.shape
       transformed_image = np.zeros_like(Is)

    inverse = np.linalg.inv(A) + b
    for i in range(Hd):
        for j in range(Wd):
            
            original = np.dot(inverse, [j / Wd * col, i / Hd * row])
            x, y = original[0], original[1]

            if 0 <= x < col-1 and 0 <= y < row-1:
              transformed_image[i, j] = Is[int(round(y)), int(round(x))]
    return transformed_image[50:200,50:200]

Is = misc.face()
Is = np.asarray(Is)
Hd,Wd = 200,200

A = .25*np.eye(2) + np.random.normal(size=(2, 2))
A = [[1.5,0.3], [0.1,1.5]]
b = np.array([[0],[0]])

(x1, y1), (x2, y2), (x3, y3), (x4, y4) = (0,0), (Is.shape[0],0), (0,Is.shape[1]), (Is.shape[0], Is.shape[1])
(x1_,y1_), (x2_,y2_), (x3_,y3_), (x4_,y4_) = (0,0), (Wd,0), (0,Hd), (Wd,Hd)
Qs, Qd = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)], [(x1_,y1_), (x2_,y2_), (x3_,y3_), (x4_,y4_)]

res = recover_projective(Qs, Qd)
print(f"Parametri projekcijske transformacije su: {res}")

A,b = recover_affine_diamond(Is.shape[0],Is.shape[1], Hd,Wd)

Id1 = affine_nn(Is, A,b, Hd, Wd)
Id2 = affine_bilin(Is, A,b, Hd, Wd)
print(f"Standardna devijacija > {np.std(Id2-Id1)}")

fig = plt.figure()
if len(Is.shape)==2: plt.gray()
for i,im in enumerate([Is, Id1, Id2]):
  fig.add_subplot(1,3, i+1)
  plt.imshow(im.astype(int))
plt.show()