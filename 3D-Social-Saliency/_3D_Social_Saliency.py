
# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
import argparse
import numpy as np
#import quaternion
from pyquaternion import Quaternion
import math
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

openPoseDirectory = 'C:\\Toolkits\\OpenPose'
openPoseBuildDirectory = openPoseDirectory + '\\build'
openPosePythonDirectory = openPoseBuildDirectory + '\\python\\openpose\\Release'
openPoseDLLDirectory = openPoseBuildDirectory + '\\x64\\Release'
openPoseBinDirectory = openPoseBuildDirectory + '\\bin'
openPoseModelsLocation = openPoseDirectory + '\\models\\'
undistortedImagesDirectory = 'E:\\AML\\Data\\boat_data\\boat_1fps_200s'
imageSet0000 = undistortedImagesDirectory + '\\00012000'
frameNumber = '\\00012000'
imagePrefix = '\\image\\image'
numberofDigits = 7
imageSuffix = '.jpg'
numberOfCameras = 12
firstImage = imageSet0000 + '\\image\\image0000006.jpg'

cameraIntrinsicsFileLocation = 'E:\\AML\\Data\\boat_data\\boat_1fps_200s\\calibration\\intrinsic_z.txt'
cameraExtrinsicsFileLocation = 'E:\\AML\\Data\\boat_data\\boat_1fps_200s\\calibration\\camera_z.txt'

calibrationDirectory = undistortedImagesDirectory + '\\\calibration'

globalAtHome = False
EPIPOLAR_MATCHING = False
def HomeComputer(home):
    if (home):
        undistortedImagesDirectory = 'C:\\Users\\Zach\\source\\repos\\ComputerVision\\3D Human Reconstruction'
        cameraIntrinsicsFileLocation = undistortedImagesDirectory + '\\intrinsic_z.txt'
        cameraExtrinsicsFileLocation = undistortedImagesDirectory + '\\camera_z.txt'
    return

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def GetFundamentalMatrix(K1, R1, t1, c1, K2, R2, t2, c2):
    #K1it = np.transpose(np.linalg.inv(K1))
    #K2i = np.linalg.inv(K2)
    #R1i = np.transpose(R1)
    #skewSym = skew(t2-t1)

    #i = np.dot(K1it,R1i)
    #i = np.dot(i,skewSym)
    #i = np.dot(i,R1i)
    #i = np.dot(i,R2)
    #i = np.dot(i,K2i)

    K1it = np.transpose(np.linalg.inv(K1))
    K2i = np.linalg.inv(K2)
    R1i = np.transpose(R1)
    R2i = np.transpose(R2)
    skewSym = skew(t2-t1)

    #i = np.dot(K1it, R1i)
    #i = np.dot(i, skewSym)
    #j = np.dot(R2,K2i)
    #i = np.dot(i,j)

    i = np.dot(K1it, R1)
    i = np.dot(i,skew(c1-c2))
    i = np.dot(i,R2i)
    i = np.dot(i,K2i)

    return i

def GetCameraIntrinsics (directory, numCameras):
    cameraIntrinsics = [np.zeros((3,3)) for i in range(numCameras)]
    currentMatrix = 0
    currentRow = 0
    with open(directory) as f:
        for line in f:
            tokens = line.split()
            if tokens[0][0] == '#':
                continue
            for i in range(3):
                cameraIntrinsics[currentMatrix][currentRow,i] = float(tokens[i])
            currentRow += 1
            if (currentRow == 3):
                currentMatrix += 1
                currentRow = 0
    return cameraIntrinsics

def getR(Rt):
    return Rt[:3,:3]

def getT(Rt):
    return Rt[:,3]

def getC(Rt):
    return -np.dot(Rt[:3,:3],Rt[:,3])

def GetCameraExtrinsics (directory, numCameras):
    cameraExtrinsics = [np.zeros((3,4)) for i in range(numCameras)]
    currentMatrix = 0
    currentRow = 0
    readTranslate = True
    cameraFromWorld = [np.zeros(3) for i in range(numCameras)]
    with open(directory) as f:
        for line in f:
            tokens = line.split()
            if tokens[0][0] == '#':
                continue
            if readTranslate:
                for i in range(3):
                    #cameraExtrinsics[currentMatrix][i,3] = float(tokens[i])
                    cameraFromWorld[currentMatrix][i] = float(tokens[i])
                readTranslate = False
            else:
                for i in range(3):
                    cameraExtrinsics[currentMatrix][currentRow,i] = float(tokens[i])
                currentRow += 1
            if (currentRow == 3):
                #cameraExtrinsics[currentMatrix][:3,:3] = np.transpose(cameraExtrinsics[currentMatrix][:3,:3])
                worldFromCamera = -np.dot(cameraExtrinsics[currentMatrix][:3,:3], cameraFromWorld[currentMatrix])
                cameraExtrinsics[currentMatrix][:,3] = worldFromCamera
                currentMatrix += 1
                currentRow = 0
                readTranslate = True
    return cameraExtrinsics, cameraFromWorld

def GetImagesToLoad ():
    formatSpec = '0' + str(numberofDigits)
    return [undistortedImagesDirectory + frameNumber + imagePrefix + format(i,formatSpec) + imageSuffix for i in range(12)]

def loadImages(images2Load):
    firstimage = images2Load[0]
    img = cv2.imread(firstImage)
    imgs = [cv2.imread(imgstring) for imgstring in images2Load]
    return imgs


def DrawLineOnImage(img, l, lineColor = (0,255,0), lineWidth = 4):
    if (np.linalg.norm(l) < .00001):
        return
    d0 = 0; #left of image
    d1 = img.shape[1]-1; #right of image
    y0 = int(-(l[0]*d0+l[2])/l[1]);
    y1 = int(-(l[0]*d1+l[2])/l[1]);
    _, pt1, pt2 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]),(d0,y0),(d1,y1))
    cv2.line(img, pt1, pt2, lineColor, lineWidth)
    return

def DrawPointsWithConfidence (img, opTensor):
    for p in range(opTensor.shape[0]):
        for i in range(opTensor.shape[1]):
            #if not (i == 0 or i == 15 or i == 16 or i == 17 or i == 18):
            #    continue
            color = (0,255,0)

            if opTensor[p,i,2] < .70:
                color = (0,255,255)
            
            if opTensor[p,i,2] < .2:
                color = (0,0,255)

            cv2.circle(img, (opTensor[p,i,0],opTensor[p,i,1]),3,color,2)
    return


def DistanceFromLine (line, point):
    homogPoint = np.array([point[0], point[1], 1])
    proj = np.dot(line, homogPoint)
    lineNormal = np.linalg.norm(np.array([line[0],line[1]]))
    return proj / lineNormal

def DistanceBetweenTwoPoints (ptA, ptB) :
    displacement = ptB - ptA
    return math.sqrt(displacement[0]**2 + displacement[1]**2)


# fix person 1.
# pick a random image
def TwoPersonRANSAC(datums, intrinsics, extrinsics, cameraLocations):
    return

# Px is the projection matrix, px is the corresponding pixel coordinates of the same 3D point
# px should be shape 1xNx3 (body part)x(correspondence in each image)x(x,y,confidence)
def TriangulatePointsOLD(P, p1, p2, confidenceThreshold = .7):

    correspondences = np.zeros((p1.shape[0]*6,4))

    j = 0
    
    for i in range(p1.shape[0]):
        val = p1[i,2]
        if val > confidenceThreshold:
            pixel = (p1[i,0],p1[i,1],1)
            correspondences[j:j+3,:] = np.dot(skew(pixel),P[i])
        j += 3


    for i in range(p2.shape[0]):
        if p2[i,2] > confidenceThreshold:
            pixel = (p2[i,0],p2[i,1],1)
            correspondences[j:j+3,:] = np.dot(skew(pixel),P[i])
        j += 3

    u, s, vh = np.linalg.svd(correspondences, full_matrices=True)
    nullSpace = vh[-1,:]

    X = nullSpace / nullSpace[3]

    #pixelA = np.array([datums[imgIDA].poseKeypoints[personA,2,0],datums[imgIDA].poseKeypoints[personA,2,1],1])
    #    pixelB = np.array([datums[imgIDB].poseKeypoints[personB,2,0],datums[imgIDB].poseKeypoints[personB,2,1],1])
    #    pixelC = np.array([datums[imgIDC].poseKeypoints[personC,2,0],datums[imgIDC].poseKeypoints[personC,2,1],1])


    #    skewA = skew(pixelA)
    #    skewB = skew(pixelB)
    #    skewC = skew(pixelC)
        
    #    tfA = np.dot(skewA,projectiveMatrixA)
    #    tfB = np.dot(skewB,projectiveMatrixB)

    #    bigMatrix = np.vstack((tfA,tfB))

    #    u, s, vh = np.linalg.svd(bigMatrix, full_matrices=True)

    #    nullSpace = vh[-1,:]
    #    worldCoordinate = nullSpace / nullSpace[3]

    return X

def TriangulatePoints(P, p, confidenceThreshold = .7):

    correspondences = np.zeros((p.shape[0]*3,4))

    j = 0
    goodCorrespondences = 0
    for i in range(p.shape[0]):
        val = p[i,2]
        if val > confidenceThreshold:
            goodCorrespondences += 1
            pixel = (p[i,0],p[i,1],1)
            correspondences[j:j+3,:] = np.dot(skew(pixel),P[i])
        j += 3

    if goodCorrespondences < 2:
        return np.array([0,0,0,-1])

    u, s, vh = np.linalg.svd(correspondences, full_matrices=True)
    nullSpace = vh[-1,:]

    X = nullSpace / nullSpace[3]

    return X

def Triangulate2Points( Ex, In, Cfw, imA, imB, pA, pB, confidenceThreshold = .7 ):
     # geometric method:
    R1inv = np.transpose(getR(Ex[imA]))
    R2inv = np.transpose(getR(Ex[imB]))
    K1inv = np.linalg.inv(In[imA])
    K2inv = np.linalg.inv(In[imB])
    ray1 = np.dot(np.dot(R1inv, K1inv),pA)
    ray2 = np.dot(np.dot(R2inv, K2inv),pB)
    A = np.transpose(np.vstack((ray1, -ray2))) # numpy is really dumb, the rays get turned back into 1d vectors which numpy just treats as rows....
    t1 = getT(Ex[imA])
    t2 = getT(Ex[imB])
    b = Cfw[imB] - Cfw[imA] #np.dot(R1inv,t1)-np.dot(R2inv,t2)
    distance = np.linalg.lstsq(A,b)[0]
        
    X1world = distance[0] * ray1 + Cfw[imgIDA]
    X2world = distance[1] * ray2 + Cfw[imgIDB]
    XAvgWorld = X1world + ( ( X2world - X1world ) * .5 )
    return XAvgWorld

def getPoint2LineDistance( point, line ):
        pixel = (point[0],point[1],1)
        augmentedLineMagnitude = math.sqrt(line[0]**2 + line[1]**2)
        return abs(np.dot(pixel,line))/augmentedLineMagnitude

# Choose N Unique values from the set x
# fixed contains the indices of values in x that you wish to for sure include
# fixed as index false means that you want to keep specific values in x (slower)
def chooseNUnique(x, n, fixed = [], fixedAsIndex = True):
       
    choiceSize = n
    p = np.zeros(choiceSize);
    seen = [-1 for i in range(choiceSize)];
    for i in range(len(fixed)):
        seen[i] = fixed[i]
        p[i] = x[seen[i]]
        if not fixedAsIndex:
            p[i] = fixed[i]
            seen[i] = np.where(x == fixed[i])[0][0] # get the first occurrence of fixed[i]

    numFeatures = len(x);
    
    count = len(fixed);
    while count < choiceSize:
        randid = np.random.randint(0,numFeatures);

        if any([seen[i] == randid for i in range(len(seen))]): # if the value has been seen and accepted
            continue;

        seen[count] = randid;
        p[count] = x[randid];
        count = count + 1;

    return p


if __name__ == '__main__':
    #try:
        # Import Openpose (Windows/Ubuntu/OSX)
        




        if globalAtHome:
            undistortedImagesDirectory = 'C:\\Users\\Zach\\source\\repos\\ComputerVision\\3D Human Reconstruction'
            cameraIntrinsicsFileLocation = undistortedImagesDirectory + '\\intrinsic_z.txt'
            cameraExtrinsicsFileLocation = undistortedImagesDirectory + '\\camera_z.txt'






        dir_path = os.path.dirname(os.path.realpath(__file__))
        try:
            # Windows Import
            if sys.platform == "win32":
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append(openPosePythonDirectory);
                os.environ['PATH']  = os.environ['PATH'] + ';' + openPoseDLLDirectory + ';' +  openPoseBinDirectory + ';'
                import pyopenpose as op
            else:
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append('../../python');
                # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
                # sys.path.append('/usr/local/python')
                from openpose import pyopenpose as op
        except ImportError as e:
            print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?\nI\'m searching in ' + openPoseBuildDirectory)
            raise e

        # Flags
        parser = argparse.ArgumentParser()
        img1 = 'C:\\Toolkits\\OpenPose\\examples\\media\\COCO_val2014_000000000192.jpg'
        parser.add_argument("--image_path", default=firstImage, help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
        args = parser.parse_known_args()

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = openPoseModelsLocation

        # Add others in path?
        for i in range(0, len(args[1])):
            curr_item = args[1][i]
            if i != len(args[1])-1: next_item = args[1][i+1]
            else: next_item = "1"
            if "--" in curr_item and "--" in next_item:
                key = curr_item.replace('-','')
                if key not in params:  params[key] = "1"
            elif "--" in curr_item and "--" not in next_item:
                key = curr_item.replace('-','')
                if key not in params: params[key] = next_item

        # Construct it from system arguments
        # op.init_argv(args[1])
        # oppython = op.OpenposePython()

        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        ## Process Image
        #datum = op.Datum()
        #imageToProcess = cv2.imread(args[0].image_path)
        #datum.cvInputData = imageToProcess
        #opWrapper.emplaceAndPop([datum])

        ## Display Image
        ##np.set_printoptions(threshold=sys.maxsize)
        #print("Body keypoints: \n" + str(datum.poseKeypoints))
        #print("Body keypoints shape: " + str(datum.poseKeypoints.shape))
        #cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)

        ##newImage = imageToProcess.copy()
        #firstPerson = datum.poseKeypoints[0,:,:]
        ##logicalindex = [False for i in range(firstPerson.shape[1])]
        ##logicalindex[0] = True
        ##logicalindex[15] = True
        ##logicalindex[16] = True
        ##logicalindex[17] = True
        ##logicalindex[18] = True
        ##for p in range(datum.poseKeypoints.shape[0]):
        ##    for i in range(datum.poseKeypoints.shape[1]):
        ##        #if not (i == 0 or i == 15 or i == 16 or i == 17 or i == 18):
        ##        #    continue
        ##        color = (0,255,0)

        ##        if datum.poseKeypoints[p,i,2] < .70:
        ##            color = (0,255,255)
            
        ##        if datum.poseKeypoints[p,i,2] < .2:
        ##            color = (0,0,255)

        ##        cv2.circle(newImage, (datum.poseKeypoints[p,i,0],datum.poseKeypoints[p,i,1]),3,color,2)

        ##DrawPointsWithConfidence(newImage,datum.poseKeypoints)
        ##DrawLineOnImage(newImage,[1,-1,0])
        ##cv2.imshow("Confidence Points", newImage)

        testbodypartID = 4
        imagesToLoad = GetImagesToLoad()
        datums = [op.Datum() for i in range(numberOfCameras)]
        imagesToProcess = loadImages(imagesToLoad)
        for i in range(numberOfCameras):
            datums[i].cvInputData = imagesToProcess[i]
            opWrapper.emplaceAndPop([datums[i]])
        
        imgIDA = 2
        imgIDB = 6
        imgIDC = 4
        imgIDD = 8
        #newImage = imagesToProcess[imgIDA].copy()
        #DrawPointsWithConfidence(newImage,datums[imgIDA].poseKeypoints)
        #cv2.imshow("Confidence Points", newImage)


        cameraIntrinsics = GetCameraIntrinsics(cameraIntrinsicsFileLocation, numberOfCameras)
        cameraExtrinsics, cameraFromWorld = GetCameraExtrinsics(cameraExtrinsicsFileLocation, numberOfCameras)
        cameraProjs = [np.zeros((3,4)) for i in range(numberOfCameras)]
        for i in range(numberOfCameras):
            cameraProjs[i] = np.dot(cameraIntrinsics[i], cameraExtrinsics[i])

        #for p in range(datums[imgIDA].poseKeypoints.shape[0]//4):
            #for i in range(datums[imgIDA].poseKeypoints.shape[1]):
                #if not (i == 0 or i == 15 or i == 16 or i == 17 or i == 18):
                #    continue
                #print(p)\
        personA = 3
        personB = 1
        personC = 4

        ## ground Truth
        #color = (255,255,0)
        #cv2.circle(imagesToProcess[imgIDA], (datums[imgIDA].poseKeypoints[personA,2,0],datums[imgIDA].poseKeypoints[personA,2,1]),3,color,2)
        #cv2.circle(imagesToProcess[imgIDB], (datums[imgIDB].poseKeypoints[personB,2,0],datums[imgIDB].poseKeypoints[personB,2,1]),3,color,2)
        #cv2.circle(imagesToProcess[imgIDC], (datums[imgIDC].poseKeypoints[personC,2,0],datums[imgIDC].poseKeypoints[personC,2,1]),3,color,2)

        projectiveMatrix0 = np.dot(cameraIntrinsics[0],cameraExtrinsics[0])
        projectiveMatrixA = np.dot(cameraIntrinsics[imgIDA],cameraExtrinsics[imgIDA])
        projectiveMatrixB = np.dot(cameraIntrinsics[imgIDB],cameraExtrinsics[imgIDB])
        projectiveMatrixC = np.dot(cameraIntrinsics[imgIDC],cameraExtrinsics[imgIDC])
        ci = cameraIntrinsics[imgIDD]
        ce = cameraExtrinsics[imgIDD]
        projectiveMatrixD = np.dot(ci,ce)
        
        pixelA = np.array([datums[imgIDA].poseKeypoints[personA,2,0],datums[imgIDA].poseKeypoints[personA,2,1],1])
        pixelB = np.array([datums[imgIDB].poseKeypoints[personB,2,0],datums[imgIDB].poseKeypoints[personB,2,1],1])
        pixelC = np.array([datums[imgIDC].poseKeypoints[personC,2,0],datums[imgIDC].poseKeypoints[personC,2,1],1])


        skewA = skew(pixelA)
        skewB = skew(pixelB)
        skewC = skew(pixelC)
        
        tfA = np.dot(skewA,projectiveMatrixA)
        tfB = np.dot(skewB,projectiveMatrixB)

        bigMatrix = np.vstack((tfA,tfB))

        u, s, vh = np.linalg.svd(bigMatrix, full_matrices=True)

        nullSpace = vh[-1,:]
        worldCoordinate = nullSpace / nullSpace[3]

        
        t1 = np.zeros((2,3))
        t2 = np.zeros((2,3))
        t1[0] = pixelA
        t2[1] = pixelB
        worldCoordinate2 = TriangulatePointsOLD(np.array([projectiveMatrixA,projectiveMatrixB]),t1,t2)


        # geometric method:
        R1inv = np.transpose(getR(cameraExtrinsics[imgIDA]))
        R2inv = np.transpose(getR(cameraExtrinsics[imgIDB]))
        K1inv = np.linalg.inv(cameraIntrinsics[imgIDA])
        K2inv = np.linalg.inv(cameraIntrinsics[imgIDB])
        ray1 = np.dot(np.dot(R1inv, K1inv),pixelA)
        ray2 = np.dot(np.dot(R2inv, K2inv),pixelB)
        A = np.transpose(np.vstack((ray1, -ray2))) # numpy is really dumb, the rays get turned back into 1d vectors which numpy just treats as rows....
        t1 = getT(cameraExtrinsics[imgIDA])
        t2 = getT(cameraExtrinsics[imgIDB])
        b = cameraFromWorld[imgIDB] - cameraFromWorld[imgIDA] #np.dot(R1inv,t1)-np.dot(R2inv,t2)
        distance = np.linalg.lstsq(A,b)[0]
        
        X1world = distance[0] * ray1 + cameraFromWorld[imgIDA] #- np.dot(R1inv,t1)
        X2world = distance[1] * ray2 + cameraFromWorld[imgIDB] #- np.dot(R2inv,t2)
        XAvgWorld = X1world + ((X2world-X1world)*.5)
        XAvgWorld2 = (worldCoordinate[:3] + X1world + X2world) * (1/3)





        if not EPIPOLAR_MATCHING:
            print('RANSAC METHOD')

            initialPerson = 0#personA
            initialImage = imgIDA

            candidatesInImages = [None] * numberOfCameras # holds logical arrays for each camera where a True/1 represents a person is still a candidate and False/0 means they have been eliminated (either associating it with another person or by removing them entirely)
            for i in range(numberOfCameras):
                candidatesInImages[i] = np.ones(datums[i].poseKeypoints.shape[0], dtype=bool) # all people for datum i

            candidateImages = np.ones(numberOfCameras, dtype=bool)
            candidateImages[initialImage] = False
            #candidateImages[imgIDB] = True
            #candidateImages[imgIDC] = True
            

            #randomImages = chooseNUnique( candidateImagesIdx, 2, [initialImage], False ) # always keep the index at [imgIDA]
            RANSAC_ITERATIONS = 500

            candidateImages[initialImage] = False # don't choose the original image for triangulation
            candidateImagesIdx = np.where( candidateImages == True )[0] # index buffer

            bestInliers = 0
            bestPerson = -1
            bestImage = -1
            confidenceThreshold = .7
            ransacDistanceThreshold = 15


            for r in range( RANSAC_ITERATIONS ):
                inliers = 0
                randomImage = int(chooseNUnique( candidateImagesIdx, 1 )[0])

                candidatesInRandomImageIdx = np.where( candidatesInImages[randomImage] == True )[0] # index buffer to people
                randomPerson = int(chooseNUnique( candidatesInRandomImageIdx, 1 )[0])

                for bodypartID in range( datums[initialImage].poseKeypoints.shape[1] ): # for every body part
                    bodypartA = datums[initialImage].poseKeypoints[initialPerson,bodypartID]
                    bodypartB = datums[randomImage].poseKeypoints[randomPerson,bodypartID]
                    if bodypartA[2] < confidenceThreshold or bodypartB[2] < confidenceThreshold:
                        continue
                    pixelA = ( bodypartA[0], bodypartA[1], 1 )
                    pixelB = ( bodypartB[0], bodypartB[1], 1 )
                    X = Triangulate2Points( cameraExtrinsics, cameraIntrinsics, cameraFromWorld, initialImage, randomImage, pixelA, pixelB )
                    
                    imagesLeft = np.array(candidateImages)
                    imagesLeft[randomImage] = False # don't include the random image in the reprojection

                    for i in np.where(imagesLeft == True)[0]: # in every other image
                        pix = np.dot(cameraProjs[i], np.array([X[0],X[1],X[2],1]))
                        pix /= pix[2]

                        for p in np.where( candidatesInImages[i] == True )[0]: # for every person remaining in the image
                            if datums[i].poseKeypoints[p,bodypartID,2] < confidenceThreshold:
                                continue
                            otherpix = ( datums[i].poseKeypoints[p,bodypartID,0], datums[i].poseKeypoints[p,bodypartID,1], 1) 
                            distance = np.linalg.norm( otherpix - pix )
                            if distance < ransacDistanceThreshold:
                                inliers += 1

                if inliers > bestInliers:
                    bestInliers = inliers
                    bestPerson = randomPerson
                    bestImage = randomImage


                    
            color = (0,255,0)
            for bodypartID in range( datums[initialImage].poseKeypoints.shape[1] ):
                bodypartPixelA = np.array([datums[initialImage].poseKeypoints[initialPerson,bodypartID,0], datums[initialImage].poseKeypoints[initialPerson,bodypartID,1], 1])
                bodypartPixelB = np.array([datums[bestImage].poseKeypoints[bestPerson,bodypartID,0], datums[bestImage].poseKeypoints[bestPerson,bodypartID,1], 1])
                cv2.circle(imagesToProcess[initialImage], (int(bodypartPixelA[0]),int(bodypartPixelA[1])),3,color,2)
                cv2.circle(imagesToProcess[bestImage], (int(bodypartPixelB[0]),int(bodypartPixelB[1])),3,color,2)

            cv2.namedWindow('Initial Image', cv2.WINDOW_NORMAL)
            cv2.imshow('Initial Image',imagesToProcess[initialImage])
            cv2.resizeWindow('Initial Image', imagesToProcess[initialImage].shape[1]//2, imagesToProcess[initialImage].shape[0]//2)

            cv2.namedWindow('B', cv2.WINDOW_NORMAL)
            cv2.imshow('B',imagesToProcess[bestImage])
            cv2.resizeWindow('B', imagesToProcess[bestImage].shape[1]//2, imagesToProcess[bestImage].shape[0]//2)
            cv2.waitKey()



        else:
            # Let's find and triangulate a person in another image given a person in the current image!
            initialPerson = 0#personA
            initialImage = 6#imgIDA


            #keyPointsInEachImage = np.zeros((numberOfCameras, datums[initialImage].poseKeypoints.shape[1], 3)) #  bodyparts (25) x images (matches) x 3 points  # allows for indexing by x[i] means all pixels from the each image corresponding to bodypart i
            keyPointsInEachImage = np.zeros((datums[initialImage].poseKeypoints.shape[1], numberOfCameras, 3)) #  bodyparts (25) x images (matches) x 3 points  # allows for indexing by x[i] means all pixels from the each image corresponding to bodypart i
            keyPointsInEachImage[:,initialImage,:] = datums[initialImage].poseKeypoints[initialPerson]
            matchThreshold = 8 # must have at least this many points correllated with the original person to work TODO: this is bad if the original person had less than 'matchThreshold' points... need some way to choose people without seeds
            matchRatioThreshold = .6 # the percentage difference between the second best and first best must be less than this
        
            confidenceThreshold = .7 # points with less confidence than this are discarded from matching
            distThreshold = 20 # if it's within distThreshold pixels it's a good match
            triangulationConfidenceThreshold = .7 # points with less confidence than this are discarded from triangulation


            for otherImage in range(numberOfCameras):
                if otherImage == initialImage:
                    continue
        
                fundMatrixB = GetFundamentalMatrix(cameraIntrinsics[initialImage],cameraExtrinsics[initialImage][:3,:3],cameraExtrinsics[initialImage][:3,3], cameraFromWorld[initialImage], cameraIntrinsics[otherImage],cameraExtrinsics[otherImage][:3,:3],cameraExtrinsics[otherImage][:3,3], cameraFromWorld[otherImage])
                fundTransB = np.transpose(fundMatrixB)
        

                matchCounter = np.zeros((datums[otherImage].poseKeypoints.shape[0], datums[otherImage].poseKeypoints.shape[1])) # number of people in imageB x number of bodyparts in imageB

                for bodypartID in range(datums[initialImage].poseKeypoints.shape[1]): # for all body parts
                    #bodypartID = 2
                    bodypartPixelA = np.array([datums[initialImage].poseKeypoints[initialPerson,bodypartID,0], datums[initialImage].poseKeypoints[initialPerson,bodypartID,1], 1])
                    print('checking body part ' + str(bodypartID))
                    if datums[initialImage].poseKeypoints[initialPerson,bodypartID,2] < confidenceThreshold:
                        print('not confidence enough for bodypart ' + str(bodypartID))
                        cv2.circle(imagesToProcess[initialImage], (int(bodypartPixelA[0]),int(bodypartPixelA[1])),3,(0,0,255),2)
                        continue

                    color = (0,255,0)
                    cv2.circle(imagesToProcess[initialImage], (int(bodypartPixelA[0]),int(bodypartPixelA[1])),3,color,2)
                    lineInB = np.dot(fundTransB,bodypartPixelA)
                    DrawLineOnImage( imagesToProcess[otherImage], lineInB)
        
                    print('There are ' + str(datums[otherImage].poseKeypoints.shape[0]) + ' people in image ' + str(otherImage))
                    for personIndex in range(datums[otherImage].poseKeypoints.shape[0]): # for each person in image b[initialPerson,point,0] # TODO: order of nesting should change to avoid cache misses (for each person for each body part
            
                        #DrawLineOnImage( imagesToProcess[i], lineInB )
        
                        #DrawLineOnImage( imagesToProcess[initialImage], lineInA)
                        #DrawLineOnImage( imagesToProcess[imgIDB], lineInB)
                        #DrawLineOnImage( imagesToProcess[imgIDD], lineInD)
                        if datums[otherImage].poseKeypoints[personIndex,bodypartID,2] < confidenceThreshold:
                            print('not confident enough (' + str(datums[otherImage].poseKeypoints[personIndex,bodypartID,2]) + ') for person ' + str(personIndex))
                            #cv2.circle(imagesToProcess[otherImage], (int(bodypartPixelB[0]),int(bodypartPixelB[1])),3,(0,0,255),2)
                            continue

                        bodypartPixelB = np.array([datums[otherImage].poseKeypoints[personIndex,bodypartID,0], datums[otherImage].poseKeypoints[personIndex,bodypartID,1], 1])
                        dist = getPoint2LineDistance(bodypartPixelB,lineInB)
                        if dist < distThreshold:
                            print("match at " + str(personIndex))
                            matchCounter[personIndex,bodypartID] += 1
                        print("person " + str(personIndex) + ", dist " + str(dist))
                        color = (255 * np.clip((dist/50),0,1), 255 * np.clip((1 - dist/50),0,1), 255 * np.clip((dist/50),0,1))
                        cv2.circle(imagesToProcess[otherImage], (int(bodypartPixelB[0]),int(bodypartPixelB[1])),3,color,2)
                    
                        #cv2.namedWindow('B', cv2.WINDOW_NORMAL)
                        #cv2.imshow('B',imagesToProcess[otherImage])
                        #cv2.resizeWindow('B', imagesToProcess[otherImage].shape[1]//2, imagesToProcess[otherImage].shape[0]//2)
                        #cv2.waitKey();

                matchesPerPerson = np.sum(matchCounter,axis = 1)
                matchingPersonIndex = np.argmax(matchesPerPerson)
                print('The best match in image ' + str(otherImage) + ' is person ' + str(matchingPersonIndex) + ' with ' + str(matchesPerPerson[matchingPersonIndex]) + ' matches.') # Preson B was ' + str(personB))
            
                #datumA = datums[initialImage].poseKeypoints[initialPerson]
                datumB = datums[otherImage].poseKeypoints[matchingPersonIndex]

                belowMatchThreshold = matchesPerPerson[matchingPersonIndex] < matchThreshold
                sortedMatchesPerPerson = matchesPerPerson[np.argsort(matchesPerPerson)]
                matchRatio = sortedMatchesPerPerson[-2]/sortedMatchesPerPerson[-1]
                tooSimilarMatches = matchRatio > matchRatioThreshold

                for i, pointB in zip(range(datumB.shape[0]), datumB):
                    #if i == 4 or i == 3:
                    #cv2.circle(imagesToProcess[initialImage], (int(pointA[0]),int(pointA[1])),4,(255,255,0),4)
                
                    if belowMatchThreshold or tooSimilarMatches:
                        cv2.circle(imagesToProcess[otherImage], (int(pointB[0]),int(pointB[1])),4,(0,0,255),4)
                    elif  pointB[2] < confidenceThreshold:
                        cv2.circle(imagesToProcess[otherImage], (int(pointB[0]),int(pointB[1])),4,(0,255,255),4)
                    else:
                        cv2.circle(imagesToProcess[otherImage], (int(pointB[0]),int(pointB[1])),4,(255,255,0),4)

                cv2.namedWindow('B', cv2.WINDOW_NORMAL)
                cv2.imshow('B',imagesToProcess[otherImage])
                cv2.resizeWindow('B', imagesToProcess[otherImage].shape[1]//2, imagesToProcess[otherImage].shape[0]//2)
                cv2.waitKey();


                if belowMatchThreshold or tooSimilarMatches:
                    continue
                else:
                    keyPointsInEachImage[:,otherImage,:] = datums[otherImage].poseKeypoints[matchingPersonIndex]


        
            # END FOR EACH IMAGE    


                    
            links = np.array([[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[9,10],[10,11],[11,22],[22,23],[11,24],[8,12],[12,13],[13,14],[14,19],[19,20],[14,21],[0,15],[15,17],[0,16],[16,18]])
            reconstructedPoints = np.zeros((datums[initialImage].poseKeypoints.shape[1],4))
        
            #relaxedConfidenceThreshold = .6
            for i in range(reconstructedPoints.shape[0]): # for each body part
                print('Reconstructing body part ' + str(i) + ' with confidence better than ' + str(triangulationConfidenceThreshold) + '...')
                #while True:
                reconstructedPoints[i] = TriangulatePoints(cameraProjs, keyPointsInEachImage[i], triangulationConfidenceThreshold)
            print('done reconstructing')
        

            #links = np.array([[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[9,10],[10,11],[11,22],[22,23],[11,24],[8,12],[12,13],[13,14],[14,19],[19,20],[14,21],[0,15],[15,17],[0,16],[16,18]])
            #reconstructedPoints = np.zeros((datums[imgIDA].poseKeypoints.shape[1],4))
        
            #relaxedConfidenceThreshold = .6
            #for i in range(reconstructedPoints.shape[0]):
            #    confidenceA = datums[imgIDA].poseKeypoints[initialPerson,i,2]
            #    confidenceB = datums[otherImage].poseKeypoints[matchingPersonIndex,i,2]
            #    if confidenceA > relaxedConfidenceThreshold and confidenceB > relaxedConfidenceThreshold:
            #        print('Reconstructing body part ' + str(i) + ' with confidence ' + str(confidenceA) + ' and ' + str(confidenceB) + '...')
            #        t1 = np.zeros((2,3))
            #        t2 = np.zeros((2,3))
            #        t1[0] = np.array([datums[imgIDA].poseKeypoints[initialPerson,i,0], datums[imgIDA].poseKeypoints[initialPerson,i,1], 1])
            #        t2[1] = np.array([datums[otherImage].poseKeypoints[matchingPersonIndex,i,0], datums[otherImage].poseKeypoints[matchingPersonIndex,i,1], 1])
            #        reconstructedPoints[i] = TriangulatePoints(np.array([projectiveMatrixA,projectiveMatrixB]),t1,t2)
            #    else:
            #        reconstructedPoints[i,3] = -1 # no reconstruction
            #print('done reconstructing')
        


            for i, point in zip(range(reconstructedPoints.shape[0]),reconstructedPoints):
                if point[3] < 0:
                    continue
                projectedPoint = np.dot(cameraProjs[initialImage],point)
                projectedPoint = projectedPoint / projectedPoint[2]
                color = (0,255,255)
                cv2.circle(imagesToProcess[initialImage], (int(projectedPoint[0]),int(projectedPoint[1])),3,color,3)

           #for i, point in zip(range(reconstructedPoints.shape[0]),reconstructedPoints):
           #     projectedPointA = np.dot(projectiveMatrixA,point)
           #     projectedPointA = projectedPointA / projectedPointA[2]
           #     projectedPointB = np.dot(projectiveMatrixB,point)
           #     projectedPointB = projectedPointB / projectedPointB[2]
           #     color = (0,255,255)
           #     if i == 4:
           #         color = (0,0,255)
           #     cv2.circle(imagesToProcess[initialImage], (int(projectedPointA[0]),int(projectedPointA[1])),3,color,3)
           #     cv2.circle(imagesToProcess[otherImage], (int(projectedPointB[0]),int(projectedPointB[1])),3,color,3)

        
            cv2.namedWindow('Initial Image', cv2.WINDOW_NORMAL)
            cv2.imshow('Initial Image',imagesToProcess[initialImage])
            cv2.resizeWindow('Initial Image', imagesToProcess[initialImage].shape[1]//2, imagesToProcess[initialImage].shape[0]//2)

            #cv2.namedWindow('B', cv2.WINDOW_NORMAL)
            #cv2.imshow('B',imagesToProcess[otherImage])
            #cv2.resizeWindow('B', imagesToProcess[otherImage].shape[1]//2, imagesToProcess[otherImage].shape[0]//2)
            cv2.waitKey();
        
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            #ax.set_zlim3d(0, 5)
            #ax.set_ylim3d(0, 5)
            #ax.set_xlim3d(-5, 0) 
            #plt.gca().set_aspect('equal', adjustable='box')
            for link in links:
                x0 = reconstructedPoints[link[0]]
                x1 = reconstructedPoints[link[1]]
                if x0[3] < 0 or x1[3] < 0: # don't reconstruct invalid points
                    continue
                ax.plot3D((x0[0],x1[0]),(x0[2],x1[2]),(-x0[1],-x1[1]), 'gray')

            reconstructedPoints = reconstructedPoints[reconstructedPoints[:, 3] != -1]
            ax.scatter(reconstructedPoints[:,0],reconstructedPoints[:,2],-reconstructedPoints[:,1], cmap= 'Greens')
            axisEqual3D(ax)
            plt.show()
        # END EPIPOLAR_MATCHING

        

#        cv2.waitKey()
#    #except Exception as e:
#    #    print(e)
#    #    sys.exit(-1)
