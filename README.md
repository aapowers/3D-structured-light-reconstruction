# 3D-structured-light-reconstruction

CS 117 Project Report - Alissa Powers - 6/13/2017

## Project Overview

### Introduction

Our eyes are actually like two cameras viewing the world through a stereo setup. This one of the main things that allows us to percieve the world in 3D. We can use information from this stereo system in our skull, such as how far an object shifts when we view it through each eye, to estimate an object’s depth fairly accurately.
Given this knowledge, it seems like it should be easy for a computer to repeat the process that our brains do to estimate the 3D location of points from images taken in stereo.
Unfortunately, it is a little more difficult than that. Our brains already have certain information, like how far apart our eyes are, and are very good at proccesses like finding correspondences between images. From our left eye to our right eye, we don’t ofen confuse a black cat that is right in front of us with a black rock somewhere else in the image. Computers require algorithms in order to replicate these processes.
One model that has become useful for assisting computers with correspondence bewtween images is structured illumination. Structured illumination projects a known pattern onto the object that is being photographed. This makes each part of the object unique, so that the computer can easily find points that match up between images from the two cameras, rather than having an object that is all the same color and texture, with many points that appear to match up.
My goal for this project was to build a software pipeline that converts 2D stereo images taken using structured illumination into 3D models.

### Details

There are a few steps in the software pipeline:

	1.	load in the images
	2.	solve for the camera calibration information
	3.	subtract the background from images with the object
	4.	decode the images using the structured illumination
	5.	reconstruct the images into 3D points using triangulation
	6.	create a mesh from the 3D points
  
This pipeline is run multiple times on images taken of the object from different views, so we end up with multiple meshes. Each mesh is only one side of the object. We then must align the meshes and combine them into one mesh that makes up the entire object.
We mostly were given or already wrote the software pipeline steps in Matlab earlier in the quarter. However, I wanted to challenge myself to apply what I learned from the Matlab assignments to a different programming language. I decided to re-build the entire pipeline in Python, making use of libraries such as OpenCV, SciPy, and numpy.
I then decided to use Meshlab align and combine the meshes. Meshlab has a built-in Iterative Closest Point (ICP) algorithm and Poisson Surface Reconstruction algorithm. Iterative Closest Point is an algorithm that iteratively transforms a point cloud to closely match another point cloud. One cloud is held constant while the second is given a rotation and translation to make each point have a minimal distance to its closest corresponding point in the first point cloud. This is done iteratively, improving each time to minimize the error. This results in a rotation and translation matrix that align the meshes. Poisson Surface Reconstruction is an algorithm that uses the surface normals of an object to find the gradient of the implicit function that defines the surface. It spans an octree, improving resolution with each new depth it explores. This results in a connected, triangulated, watertight mesh of the object.
After using Meshlab to align and combine the meshes, I save the resulting mesh to a .ply file.

## Data Sets

### Description

I used default images of a mannequin provided by Dr. Fowlkes for this assignment. These images contained the following:

	16 calibration images
	  8 images of a checkerboard from different angles taken by the left camera
	  8 images of a checkerboard from different angles taken by the right camera
	7 sets of 84 images
	  20 images of the mannequin with projected light taken by the left camera
	  20 images of the mannequin with the inverse projected light taken by the left camera
	  1 rgb image of the object taken by the left camera
	  1 rbg image of teh background taken by the left camera
	  20 images of the mannequin with projected light taken by the right camera
	  20 images of the mannequin with the inverse projected light taken by the right camera
	  1 rgb image of the object taken by the right camera
	  1 rbg image of teh background taken by the right camera

#### Examples: 

￼![Calibration and Structured Illumination Examples](images/CalibrationAndStructuredLightExamples.png?raw=true " Calibration Examples")
￼![RGB Examples](images/RBGImageExamples.png?raw=true "RGB Examples")

## Algorithms

Because I re-built the pipeline, I both implemented functions and used built-in functions from OpenCV. I show examples of some of these below.
I ended up building many of the functions from scratch when I discovered that the OpenCV functions were often slightly different from what I needed in ways that made it easier to build myself. However, when I did use OpenCV functions, I still had to build my own functions around them as the input and output were often different than what I required and I needed to process or format the data before and after each function call.

### From Scratch Implementations

#### decode

Here is an example of decode, a function I built to get the graycode image from the set of images that were taken using structured illumination. The graycode image is what allows the computer to easily find correspondences in images. This function also creates a mask that can be applied to filter out pixels that were distorted or outside the reach of the structured illumination (and therefore outside the object that we are trying to reconstruct).
def decode(image_prefix, start, stop, thresh, undistort, camera):

    image_size = None
    nbits = (stop - start) + 1
    print('decoding {} bit code'.format(nbits))

    image = cv2.imread(image_prefix + '01.jpg')
    image = undistort(image, camera)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_size = image.shape[::-1]

    # intitialize arrays
    graycode_images = []
    goodpixels = np.ones(image.shape, dtype='int32')

    i = 0
    for bit in range(start, stop + 1):
        image = cv2.imread(image_prefix + '{0:02d}'.format(bit) + '.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.normalize(image, dst=image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image = undistort(image, camera)

        image_inverse = cv2.imread(image_prefix + '{0:02d}'.format(bit) + '_i.jpg')
        image_inverse = cv2.cvtColor(image_inverse, cv2.COLOR_BGR2GRAY)
        image_inverse = cv2.normalize(image_inverse, dst=image_inverse, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                      dtype=cv2.CV_32F)
        image_inverse = undistort(image_inverse, camera)

        # build the graycode pattern image
        graycode_image = np.zeros(image.shape, dtype='int32')
        graycode_image[image > image_inverse] = 1
        if i == 0:
            graycode_images.append(np.copy(graycode_image))
        else:
            graycode_images.append(np.bitwise_xor(graycode_images[i - 1], graycode_image))

        # remove bad pixels from mesh
        absdiff = cv2.absdiff(image, image_inverse)
        goodpixels[absdiff < thresh] = 0

    C = np.zeros_like(image, dtype='int32')
    for b in range(0, nbits):
    C = C | (graycode_images[b] << b)

    return C, goodpixels

#### remove_long_edges

Here is an example of remove_long_edges, which is an algorithm I implemented to clean the mesh. The mannequin appeared to be hiding behind large triangles, so I removed triangles with edges longer than a threshold.
def remove_long_edges(tri, X):

    thresh = 1.25
    new_tri =[]
    for i in range(tri.simplices.shape[0]):
        point1 = tri.simplices[i, 0]
        point2 = tri.simplices[i, 1]
        point3 = tri.simplices[i, 2]
        dist = max(np.linalg.norm(X[point1, :]-X[point2, :]),
                   np.linalg.norm(X[point1, :] - X[point3, :]),
                   np.linalg.norm(X[point2, :] - X[point3, :]))
        if dist < thresh:
            new_tri.append(tri.simplices[i, :])

    return np.array(new_tri)
    
### Built-In Implementations

OpenCV contains functions that help with the pipeline, such as findChessboardCorners(), stereoCalibrate(), and stereoRectify().

#### find_corners

Here is an example of a function I wrote that makes use of findChessboardCorners(). I had to build a function that called this built-in function, because I needed to format the data before calling the function and do it for each image for each camera.
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

        # if both were found then let's save the points
        if left_chessboard_found and right_chessboard_found:
            object_points.append(objp)
            left_image_points.append(left_corners)
            right_image_points.append(right_corners)
        else:
            print("there was an error while finding corners for image (#{0:02d})".format(i))

    return left_image_points, right_image_points, object_points, image_size

## Results

### Description

The pipeline provided fairly good meshes from each view of the mannequin. Unfortunately they are each distorted slightly to a point where they do not align well. When I used the Iterative Closest Point (ICP) algorithm in Meshlab, the algined scans looked more like a colorful sea monster. To view the results I used the Poisson Surface Reconstruction algorithm in Meshlab and the mannequin ended up looking more like a mandrake.

#### Examples: 
￼
￼￼![Mesh Examples](images/MeshExamples.png?raw=true "Mesh Examples")
￼
### After ICP in Meshlab

￼￼![Mesh After ICP](images/AfterICP.png?raw=true "Mesh After ICP")
￼
### After Poisson in Meshlab

￼￼![Mesh After Poisson](images/AfterPoisson.png?raw=true "Mesh After Poisson")

## Assessment and Evaluation

As we can see from the results, it is not quite as easy for a computer algorithm to get a 3D object from 2D images as it is for our brains. Even with the added help of structured light, it still can fail quite horribly. One of the major limitations is calculating calibration. When the calibration is even slightly off, it causes the meshes to distort so that they cannot align nicely to form one complete object.
In general, I believe that the correspondence calculations were accurate. Each individual mesh looks very much like the image of the mannequin for that set, and saving an image that highlight some of the correspondences found shows that they do match up pretty well.
￼
## Sample of Correspondences

I believe that the main cause of the distorted final product was the calibration calculation and undistortion of the images. Once I calculated the camera calibration, I used the rotation and translation of the cameras to undistort the images before processing them. This differed from the Matlab solution that I did earlier in the quarter. In Matlab I applied the rotation and translation to the points before and after triangulating. In Python, I use the rotation and triangulation to undistort the image before decoding. This difference, added to an imperfect camera calibration calculation, contributes to the errors we see when we try to align the meshes.
Given more time, I would re-do the pictures, measuring the distance between the cameras as well as the length of the mannequin. I would use these measurements to check the calibrate calculations as well as to check the points and mesh along the way. I would also try taking out the undistorting of the images to see if triangulatePoints() does the correct transformations. If it does not, then I would write my own triangulate function in Python. I would also work more in improving the meshes. I already use a bounding box, color mask, and remove triangles with long edges before aligning the meshes, but I would add hole filling and smoothing after using Poisson to combine the aligned meshes.

## Appendix: Software

### From Scratch

	1.	bounding_box()
	2.	decode()
	3.	find_corners()
	4.	calibrate()
	5.	reconstruct()
	6.	remove_long_triangles()

### Modified From the Web

	1.	write_ply()
	2.	delaunay_triangulate()
  
### Directly from the Web

#### OpenCV

	1.	calibrateCamera()
	2.	convertPointsFromHomogeneous()
	3.	decomposeProjectionMatrix()
	4.	dilate()
	5.	findChessboardCorners()
	6.	initUndistortRectifyMap()
	7.	normalize()
	8.	remap()
	9.	stereoCalibrate()
	10.	steroRectify()
	11.	triangulatePoints()
	12.	undistort()
  
#### SciPy

	1.	Delaunay()
  
#### Meshlab
	1.	Iterative Closest Point
	2.	Poisson Surface Reconstruction
	3.	Remove Unreferenced Vertices
