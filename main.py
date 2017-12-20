import cv2
import numpy as np
import os
import pickle
from scipy.spatial import Delaunay


ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    end_header
'''

ply_tri_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    element face %(face_num)m
    property list uchar int vertex_indices
    end_header'''

###
#   write_ply(fn, verts)
#
#   takes a filename, fn and a list of vertices, verts
#   writes the vertices to the filename provided in a .ply format
###


def write_ply(fn, verts):
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f')


###
#   decode(image_prefix, start, stop, thresh)
#
#   takes an image prefix (string), start (int), stop (int), and thresh (float)
#   reads in images and their inverses in the format of the image prefix followed by a number (from start to stop)
#   loops through the images and extracts out the gray code
#   also extracts out pixels that are 'good' (over the thresh)
#   returns C, the image with each bit as a graycode value,
#   and goodpixels, the image with 0s for bad pixels and 1s for good pixels
###

def decode(image_prefix, start, stop, thresh):
    image_size = None
    nbits = (stop - start) + 1
    print('decoding {} bit code'.format(nbits))

    image = cv2.imread(image_prefix + '01.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_size = image.shape[::-1]

    graycode_images = []  # array to store gray code from image
    goodpixels = np.ones(image.shape, dtype='int32')

    i = 0
    for bit in range(start, stop+1):
        image = cv2.imread(image_prefix + '{0:02d}'.format(bit) + '.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.normalize(image, dst=image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        image_inverse = cv2.imread(image_prefix + '{0:02d}'.format(bit) + '_i.jpg')
        image_inverse = cv2.cvtColor(image_inverse, cv2.COLOR_BGR2GRAY)
        image_inverse = cv2.normalize(image_inverse, dst=image_inverse, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # get binary pattern image
        graycode_image = np.zeros(image.shape, dtype='int32')
        graycode_image[image > image_inverse] = 1
        if i == 0:
            graycode_images.append(np.copy(graycode_image))
        else:
            graycode_images.append(np.bitwise_xor(graycode_images[i-1], graycode_image))

        # remove bad pixels from mesh
        absdiff = cv2.absdiff(image, image_inverse)
        goodpixels[absdiff < thresh] = 0

    C = np.zeros_like(image, dtype='int32')
    for b in range(0, nbits):
        C = C | (graycode_images[b] << b)

    cv2.imwrite('testoutput.jpg', cv2.normalize(goodpixels, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
    cv2.imwrite('testoutput_C.jpg', cv2.normalize(C, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))

    return C, goodpixels


###
#   reconstruct(left_projection_matrix, right_projection_matrix, scandir, thresh)
#
#   takes the left and right projection matrices, a scandir to load pictures from, and thresh to throw out bad pixels
#   calls decode() on the vertically and horizontally structured light photos from the left and right cameras
#   combines the vertical and horizontal codes and applies the goodpixels mask
#   gets the pixel coordinates of the good pixels that correspond in the left and right images
#   gets the triangulated points (solves for the z values)
#   returns X (the image in pixel coordinates x,y,z)
###


def reconstruct(left_projection_matrix, right_projection_matrix, scandir, thresh):
    decode_file = './decode_file.p'

    if os.path.isfile(decode_file):
        with open(decode_file, 'rb') as file:
            R_h, R_h_good, R_v, R_v_good, L_h, L_h_good, L_v, L_v_good = pickle.load(file)
    else:
        R_h, R_h_good = decode('{}/r_'.format(scandir), 1, 10, thresh)
        R_v, R_v_good = decode('{}/r_'.format(scandir), 11, 20, thresh)
        L_h, L_h_good = decode('{}/l_'.format(scandir), 1, 10, thresh)
        L_v, L_v_good = decode('{}/l_'.format(scandir), 11, 20, thresh)

        with open(decode_file, 'wb') as file:
            pickle.dump((R_h, R_h_good, R_v, R_v_good, L_h, L_h_good, L_v, L_v_good), file)

    Rmask = cv2.bitwise_and(R_h_good, R_v_good)
    Lmask = cv2.bitwise_and(L_h_good, L_v_good)
    cv2.imwrite('rmask.jpg', cv2.normalize(Rmask, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
    cv2.imwrite('lmask.jpg', cv2.normalize(Lmask, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
    R_code = R_h | (R_v << 10)
    L_code = L_h | (L_v << 10)

    Rmask = np.array(Rmask, dtype='uint8')
    Lmask = np.array(Lmask, dtype='uint8')
    R_code_good = cv2.bitwise_and(R_code, R_code, mask=Rmask)
    L_code_good = cv2.bitwise_and(L_code, L_code, mask=Lmask)


    left_indices = [[], []]
    right_indices = [[], []]

    hashed_R_code_good = {}

    # 860, 660
    # 1560, 1360
    # filter out pixels that are outside a known bounding box
    for i in range(660, 1360):
        for j in range(860, 1560):
            value = R_code_good[i][j]
            if value == 0:
                continue
            if value not in hashed_R_code_good:
                hashed_R_code_good[value] = []
            hashed_R_code_good[value].append((i, j))

    # 860, 660
    # 1560, 1360
    # filter out pixels that are outside a known bounding box
    for i in range(660, 1360):
        for j in range(860, 1560):
            value = L_code_good[i][j]
            if value in hashed_R_code_good and value != 0:
                return_array = hashed_R_code_good[value]
                iR, jR = return_array[0]
                left_indices[0].append(i)
                left_indices[1].append(j)
                right_indices[0].append(iR)
                right_indices[1].append(jR)

    left_indices = np.array(left_indices, dtype='float32')
    right_indices = np.array(right_indices, dtype='float32')

    X = None
    for i in range(0, left_indices.shape[1], 100):
        pts = cv2.triangulatePoints(
            left_projection_matrix,
            right_projection_matrix,
            left_indices[:, i:i+100],
            right_indices[:, i:i+100]
        )
        pts = cv2.convertPointsFromHomogeneous(pts.T).reshape(-1, 3)
        if X is not None:
            X = np.vstack((pts, X))
        else:
            X = pts

    return X


###
#   calibrate(left_image_prefix, right_image_prefix, chessboard_size)
#
#   takes the left and right image prefixes, and the size of the chessboard to use for calibration
#   finds the corners in the chessboard
#   uses those to calibrate the cameras
#   gets the camera projection matrices using stereoRectify
#   returns the camera projection matrices
###


def calibrate(left_image_prefix, right_image_prefix, chessboard_size):

    corners_file = './corners.p'

    if os.path.isfile(corners_file):
        with open(corners_file, 'rb') as file:
            left_image_points, right_image_points, object_points, image_size = pickle.load(file)
    else:
        left_image_points, right_image_points, object_points, image_size = find_corners(left_image_prefix,
                                                                                        right_image_prefix,
                                                                                        chessboard_size)
        with open(corners_file, 'wb') as file:
            pickle.dump((left_image_points, right_image_points, object_points, image_size), file)

    retval, left_cam_matrix, left_cam_dist, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=object_points,
        imagePoints=left_image_points,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    retval, right_cam_matrix, right_cam_dist, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=object_points,
        imagePoints=right_image_points,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    # get camera calibrations
    return_value, left_cam_matrix, left_cam_dist, right_cam_matrix, right_cam_dist, cam_R, cam_t, bogus, nonsense = cv2.stereoCalibrate(
        object_points,
        left_image_points,
        right_image_points,
        imageSize=image_size,
        cameraMatrix1=left_cam_matrix,
        cameraMatrix2=right_cam_matrix,
        distCoeffs1=left_cam_dist,
        distCoeffs2=right_cam_dist,
        flags=cv2.CALIB_FIX_INTRINSIC
        #  flags=cv2.CALIB_USE_INTRINSIC_GUESS
    )

    left_cam_rotate, right_cam_rotate, left_projection_matrix, right_projection_matrix, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=left_cam_matrix,
        cameraMatrix2=right_cam_matrix,
        distCoeffs1=left_cam_dist,
        distCoeffs2=right_cam_dist,
        imageSize=image_size,
        R=cam_R,
        T=cam_t
    )

    img = cv2.imread('mannequin/calib/l_calib_04.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.undistort(img, left_cam_matrix, left_cam_dist)
    cv2.imwrite('l_calib_04_undistort.jpg', img)
    found, lcorners = cv2.findChessboardCorners(img, patternSize=(10, 7))

    img = cv2.imread('mannequin/calib/r_calib_04.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.undistort(img, right_cam_matrix, right_cam_dist)
    cv2.imwrite('r_calib_04_undistort.jpg', img)
    found, rcorners = cv2.findChessboardCorners(img, patternSize=(10, 7))

    points = cv2.triangulatePoints(
        left_projection_matrix,
        right_projection_matrix,
        np.array(lcorners).T.reshape(2, 70),
        np.array(rcorners).T.reshape(2, 70)
    )
    points = cv2.convertPointsFromHomogeneous(points.T).reshape(-1, 3)
    write_ply('test.ply', points)

    return left_projection_matrix, right_projection_matrix


###
#   find_corners(left_image_prefix, right_image_prefix, chessboard_size)
#
#   takes the left and right image prefixes, and the size of the chessboard to find the corners in
#   reads in the images and find the corners using findChessboardCorners
#   adds the found corners to the list
#   returns the image points of the corners found in the right and left images
#   also returns the objection points and the size of the image
###


def find_corners(left_image_prefix, right_image_prefix, chessboard_size):
    image_size = None
    left_image_points = []
    right_image_points = []
    object_points = []

    objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    for i in range(1, 9):
        # find left chessboard
        left_image = cv2.imread(left_image_prefix + '{0:02d}'.format(i) + '.jpg')
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        image_size = left_image.shape[::-1]
        left_chessboard_found, left_corners = cv2.findChessboardCorners(left_image, chessboard_size)

        # find right chessboard
        right_image = cv2.imread(right_image_prefix + '{0:02d}'.format(i) + '.jpg')
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        right_chessboard_found, right_corners = cv2.findChessboardCorners(right_image, chessboard_size)

        # if both were found then let's save them so we can manually check them
        # and let's save the points found
        if left_chessboard_found and right_chessboard_found:
            object_points.append(objp)
            left_image_points.append(left_corners)
            output_image = cv2.drawChessboardCorners(left_image, chessboard_size, left_corners, left_chessboard_found)
            cv2.imwrite('left_corners_{0:02d}.jpg'.format(i), output_image)
            right_image_points.append(right_corners)
            output_image = cv2.drawChessboardCorners(right_image, chessboard_size, right_corners, right_chessboard_found)
            cv2.imwrite('right_corners_{0:02d}.jpg'.format(i), output_image)
        else:
            print("there was an error while finding corners for image (#{0:02d})".format(i))

    return left_image_points, right_image_points, object_points, image_size


###
#   delaunay_triangulate(fn, X)
#
#   takes a filename, fn, and an array of points, X
#   triangulates using delaunay triangulation on the x and y coordinates in X
#   outputs the result in a .ply format
###


def delaunay_triangulate(fn, X):

    points = np.zeros((X.shape[0], 2))
    for i in range(X.shape[0]):
        vals = X[i, :]
        points[i, 0] = float(vals[0])
        points[i, 1] = float(vals[1])

    tri_file = './tri.p'

    if os.path.isfile(tri_file):
        with open(tri_file, 'rb') as file:
            tri = pickle.load(file)
    else:
        tri = Delaunay(points)

        with open(tri_file, 'wb') as file:
            pickle.dump(tri, file)

    M = int(tri.simplices.shape[0])
    N = X.shape[0]

    plyFile = open(fn, 'w')
    plyFile.write('ply\n')
    plyFile.write('format ascii 1.0\n')
    plyFile.write('element vertex %i\n' % N)
    plyFile.write('property float x\nproperty float y\nproperty float z\n')
    plyFile.write('element face %i\n' % M)
    plyFile.write('property list uchar int vertex_indices\n')
    plyFile.write('end_header\n')

    # Output the points to the PLY file
    for i in range(N):
        vals = X[i, :]
        # x y z
        xyz = [float(a) for a in vals[0:3]]
        xyz[2] = -xyz[2]  # Compensate for coordinate system
        plyFile.write("%s %s %s\n" % (tuple(xyz)))

    # Output the triangles to the PLY file
    for i in range(M):
        a, b, c = tri.simplices[i, :]
        D = np.ones((3, 3))
        D[1:, 0] = points[a, :]
        D[1:, 1] = points[b, :]
        D[1:, 2] = points[c, :]
        # Make sure the triangle faces have consistent normals
        if np.linalg.det(D) > 0:
            [a, b, c] = [c, b, a]
        triString = "3 %i %i %i\n" % (a, b, c)
        plyFile.write(triString)


###
#   main()
#
#   calibrates the cameras
#   reconstructs the points
#   uses Delaunay triangulation to find the triangles
#   outputs a .ply of the mesh
###


def main():

    camL, camR = calibrate('mannequin/calib/l_calib_', 'mannequin/calib/r_calib_', (10, 7))

    reconstruct_file = './reconstruction.p'

    if os.path.isfile(reconstruct_file):
        with open(reconstruct_file, 'rb') as file:
            X = pickle.load(file)
    else:
        X = reconstruct(camL, camR, 'mannequin/set_01', .001)

        with open(reconstruct_file, 'wb') as file:
            pickle.dump(X, file)

    delaunay_triangulate('out.ply', X)

    if os.path.isfile('out.ply'):
        print("Mesh completed. You can now view the out.ply file.")


if __name__ == '__main__':
    main()
