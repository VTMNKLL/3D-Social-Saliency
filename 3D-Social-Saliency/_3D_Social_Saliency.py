
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

def HomeComputer(home):
    if (home):
        undistortedImagesDirectory = 'C:\\Users\\Zach\\source\\repos\\ComputerVision\\3D Human Reconstruction'
        cameraIntrinsicsFileLocation = undistortedImagesDirectory + '\\intrinsic_z.txt'
        cameraExtrinsicsFileLocation = undistortedImagesDirectory + '\\camera_z.txt'
    return


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
        
        #for p in range(datums[imgIDA].poseKeypoints.shape[0]//4):
            #for i in range(datums[imgIDA].poseKeypoints.shape[1]):
                #if not (i == 0 or i == 15 or i == 16 or i == 17 or i == 18):
                #    continue
                #print(p)\
        personA = 3
        personB = 1
        personC = 4

        # ground Truth
        color = (255,255,0)
        cv2.circle(imagesToProcess[imgIDA], (datums[imgIDA].poseKeypoints[personA,2,0],datums[imgIDA].poseKeypoints[personA,2,1]),3,color,2)
        cv2.circle(imagesToProcess[imgIDB], (datums[imgIDB].poseKeypoints[personB,2,0],datums[imgIDB].poseKeypoints[personB,2,1]),3,color,2)
        cv2.circle(imagesToProcess[imgIDC], (datums[imgIDC].poseKeypoints[personC,2,0],datums[imgIDC].poseKeypoints[personC,2,1]),3,color,2)

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


        pixelGuess = np.dot(projectiveMatrixC, worldCoordinate)
        pixelGuess = pixelGuess / pixelGuess[2]
        pixGuessX1 = np.dot(projectiveMatrixC, np.hstack((X1world,1)))
        pixGuessX1 = pixGuessX1 / pixGuessX1[2]
        pixGuessX2 = np.dot(projectiveMatrixC, np.hstack((X2world,1)))
        pixGuessX2 = pixGuessX2 / pixGuessX2[2]
        pixGuessXA = np.dot(projectiveMatrixC, np.hstack((XAvgWorld,1)))
        pixGuessXA = pixGuessXA / pixGuessXA[2]
        pixGuessXA2 = np.dot(projectiveMatrixC, np.hstack((XAvgWorld2,1)))
        pixGuessXA2 = pixGuessXA2 / pixGuessXA2[2]
        color = (0,0,255)
        cv2.circle(imagesToProcess[imgIDC], (int(pixelGuess[0]),int(pixelGuess[1])),3,color,2)
        color = (0,255,255)
        cv2.circle(imagesToProcess[imgIDC], (int(pixGuessX1[0]),int(pixGuessX1[1])),3,color,2)
        color = (255,255,255)
        cv2.circle(imagesToProcess[imgIDC], (int(pixGuessX2[0]),int(pixGuessX2[1])),3,color,2)
        color = (255,0,255)
        cv2.circle(imagesToProcess[imgIDC], (int(pixGuessXA[0]),int(pixGuessXA[1])),3,color,2)
        color = (0,255,0)
        cv2.circle(imagesToProcess[imgIDC], (int(pixGuessXA2[0]),int(pixGuessXA2[1])),3,color,2)
        #color = (255,255,0)
        #cv2.circle(imagesToProcess[imgIDC], (int(pixelC[0]),int(pixelC[1])),3,color,2)


        pixelGuess = np.dot(projectiveMatrixA, worldCoordinate)
        pixelGuess = pixelGuess / pixelGuess[2]
        pixGuessX1 = np.dot(projectiveMatrixA, np.hstack((X1world,1)))
        pixGuessX1 = pixGuessX1 / pixGuessX1[2]
        pixGuessX2 = np.dot(projectiveMatrixA, np.hstack((X2world,1)))
        pixGuessX2 = pixGuessX2 / pixGuessX2[2]
        pixGuessXA = np.dot(projectiveMatrixA, np.hstack((XAvgWorld,1)))
        pixGuessXA = pixGuessXA / pixGuessXA[2]
        pixGuessXA2 = np.dot(projectiveMatrixA, np.hstack((XAvgWorld2,1)))
        pixGuessXA2 = pixGuessXA2 / pixGuessXA2[2]
        color = (0,0,255)
        cv2.circle(imagesToProcess[imgIDA], (int(pixelGuess[0]),int(pixelGuess[1])),3,color,2)
        color = (0,255,255)
        cv2.circle(imagesToProcess[imgIDA], (int(pixGuessX1[0]),int(pixGuessX1[1])),3,color,2)
        color = (255,255,255)
        cv2.circle(imagesToProcess[imgIDA], (int(pixGuessX2[0]),int(pixGuessX2[1])),3,color,2)
        color = (255,0,255)
        cv2.circle(imagesToProcess[imgIDA], (int(pixGuessXA[0]),int(pixGuessXA[1])),3,color,2)
        color = (0,255,0)
        cv2.circle(imagesToProcess[imgIDA], (int(pixGuessXA2[0]),int(pixGuessXA2[1])),3,color,2)
        #color = (255,255,0)
        #cv2.circle(imagesToProcess[imgIDA], (int(pixelA[0]),int(pixelA[1])),3,color,2)

        pixelGuess = np.dot(projectiveMatrixB, worldCoordinate)
        pixelGuess = pixelGuess / pixelGuess[2]
        pixGuessX1 = np.dot(projectiveMatrixB, np.hstack((X1world,1)))
        pixGuessX1 = pixGuessX1 / pixGuessX1[2]
        pixGuessX2 = np.dot(projectiveMatrixB, np.hstack((X2world,1)))
        pixGuessX2 = pixGuessX2 / pixGuessX2[2]
        pixGuessXA = np.dot(projectiveMatrixB, np.hstack((XAvgWorld,1)))
        pixGuessXA = pixGuessXA / pixGuessXA[2]
        pixGuessXA2 = np.dot(projectiveMatrixB, np.hstack((XAvgWorld2,1)))
        pixGuessXA2 = pixGuessXA2 / pixGuessXA2[2]
        color = (0,0,255)
        cv2.circle(imagesToProcess[imgIDB], (int(pixelGuess[0]),int(pixelGuess[1])),3,color,2)
        color = (0,255,255)
        cv2.circle(imagesToProcess[imgIDB], (int(pixGuessX1[0]),int(pixGuessX1[1])),3,color,2)
        color = (255,255,255)
        cv2.circle(imagesToProcess[imgIDB], (int(pixGuessX2[0]),int(pixGuessX2[1])),3,color,2)
        color = (255,0,255)
        cv2.circle(imagesToProcess[imgIDB], (int(pixGuessXA[0]),int(pixGuessXA[1])),3,color,2)
        color = (0,255,0)
        cv2.circle(imagesToProcess[imgIDB], (int(pixGuessXA2[0]),int(pixGuessXA2[1])),3,color,2)
        #color = (255,255,0)
        #cv2.circle(imagesToProcess[imgIDB], (int(pixelB[0]),int(pixelB[1])),3,color,2)

        pixelGuess = np.dot(projectiveMatrixD, worldCoordinate)
        pixelGuess = pixelGuess / pixelGuess[2]
        pixGuessX1 = np.dot(projectiveMatrixD, np.hstack((X1world,1)))
        pixGuessX1 = pixGuessX1 / pixGuessX1[2]
        pixGuessX2 = np.dot(projectiveMatrixD, np.hstack((X2world,1)))
        pixGuessX2 = pixGuessX2 / pixGuessX2[2]
        pixGuessXA = np.dot(projectiveMatrixD, np.hstack((XAvgWorld,1)))
        pixGuessXA = pixGuessXA / pixGuessXA[2]
        pixGuessXA2 = np.dot(projectiveMatrixD, np.hstack((XAvgWorld2,1)))
        pixGuessXA2 = pixGuessXA2 / pixGuessXA2[2]
        color = (0,0,255)
        cv2.circle(imagesToProcess[imgIDD], (int(pixelGuess[0]),int(pixelGuess[1])),3,color,2)
        color = (0,255,255)
        cv2.circle(imagesToProcess[imgIDD], (int(pixGuessX1[0]),int(pixGuessX1[1])),3,color,2)
        color = (255,255,255)
        cv2.circle(imagesToProcess[imgIDD], (int(pixGuessX2[0]),int(pixGuessX2[1])),3,color,2)
        color = (255,0,255)
        cv2.circle(imagesToProcess[imgIDD], (int(pixGuessXA[0]),int(pixGuessXA[1])),3,color,2)
        color = (0,255,0)
        cv2.circle(imagesToProcess[imgIDD], (int(pixGuessXA2[0]),int(pixGuessXA2[1])),3,color,2)


        pixelGuess = np.dot(projectiveMatrix0, worldCoordinate)
        pixelGuess = pixelGuess / pixelGuess[2]
        pixGuessX1 = np.dot(projectiveMatrix0, np.hstack((X1world,1)))
        pixGuessX1 = pixGuessX1 / pixGuessX1[2]
        pixGuessX2 = np.dot(projectiveMatrix0, np.hstack((X2world,1)))
        pixGuessX2 = pixGuessX2 / pixGuessX2[2]
        pixGuessXA = np.dot(projectiveMatrix0, np.hstack((XAvgWorld,1)))
        pixGuessXA = pixGuessXA / pixGuessXA[2]
        pixGuessXA2 = np.dot(projectiveMatrix0, np.hstack((XAvgWorld2,1)))
        pixGuessXA2 = pixGuessXA2 / pixGuessXA2[2]
        color = (0,0,255)
        cv2.circle(imagesToProcess[0], (int(pixelGuess[0]),int(pixelGuess[1])),3,color,2)
        color = (0,255,255)
        cv2.circle(imagesToProcess[0], (int(pixGuessX1[0]),int(pixGuessX1[1])),3,color,2)
        color = (255,255,255)
        cv2.circle(imagesToProcess[0], (int(pixGuessX2[0]),int(pixGuessX2[1])),3,color,2)
        color = (255,0,255)
        cv2.circle(imagesToProcess[0], (int(pixGuessXA[0]),int(pixGuessXA[1])),3,color,2)
        color = (0,255,0)
        cv2.circle(imagesToProcess[0], (int(pixGuessXA2[0]),int(pixGuessXA2[1])),3,color,2)


        #for i in range(numberOfCameras):
        #    if i == imgIDA:
        #        continue
        #    fundMatrix = GetFundamentalMatrix(cameraIntrinsics[imgIDA],cameraExtrinsics[imgIDA][:3,:3],cameraExtrinsics[imgIDA][:3,3],cameraIntrinsics[i],cameraExtrinsics[i][:3,:3],cameraExtrinsics[i][:3,3])
        #    fundTrans = np.transpose(fundMatrix)

        #    #lineInA = np.dot(fundMatrix,pixelB)
        #    lineInB = np.dot(fundTrans,pixelA)
        
        #    #DrawLineOnImage( imagesToProcess[imgIDA], lineInA)
        #    DrawLineOnImage( imagesToProcess[i], lineInB)
        #    cv2.imshow(str(i), imagesToProcess[i])
        #    #cv2.imshow('A', imagesToProcess[imgIDA])
        #    #cv2.imshow('B', imagesToProcess[imgIDB])
        #    #cv2.imshow('C', imagesToProcess[imgIDC])
        #    #cv2.imshow('D', imagesToProcess[imgIDD])
        #    #cv2.imshow('0', imagesToProcess[0])
        #    #cv2.imwrite('A.jpg', imagesToProcess[imgIDA])
        #    #cv2.imwrite('B.jpg', imagesToProcess[imgIDB])
        #    #cv2.imwrite('C.jpg', imagesToProcess[imgIDC])
        #    #cv2.imwrite('D.jpg', imagesToProcess[imgIDD])
        #    #cv2.imwrite('0.jpg', imagesToProcess[0])
        #cv2.waitKey()

        fundMatrix = GetFundamentalMatrix(cameraIntrinsics[imgIDA],cameraExtrinsics[imgIDA][:3,:3],cameraExtrinsics[imgIDA][:3,3], cameraFromWorld[imgIDA], cameraIntrinsics[imgIDB],cameraExtrinsics[imgIDB][:3,:3],cameraExtrinsics[imgIDB][:3,3], cameraFromWorld[imgIDB])
        fundTrans = np.transpose(fundMatrix)

        for point in range(datums[imgIDA].poseKeypoints.shape[1]): #[personA,2,0]:
            bodypartPixel = np.array([datums[imgIDA].poseKeypoints[personA,point,0], datums[imgIDA].poseKeypoints[personA,point,1], 1])

            if datums[imgIDA].poseKeypoints[personA,point,2] < .7:
                continue
        

            #lineInA = np.dot(fundMatrix,pixelB)
            lineInB = np.dot(fundTrans,bodypartPixel)
        
            #DrawLineOnImage( imagesToProcess[imgIDA], lineInA)
            DrawLineOnImage( imagesToProcess[imgIDB], lineInB)
            
            color = (0,255,0)
            cv2.circle(imagesToProcess[imgIDA], (int(bodypartPixel[0]),int(bodypartPixel[1])),3,color,2)
        
        cv2.imshow('A', imagesToProcess[imgIDA])
        cv2.imshow('B', imagesToProcess[imgIDB])
        #cv2.imshow('C', imagesToProcess[imgIDC])
        #cv2.imshow('D', imagesToProcess[imgIDD])
        #cv2.imshow('0', imagesToProcess[0])
        #cv2.imwrite('A.jpg', imagesToProcess[imgIDA])
        #cv2.imwrite('B.jpg', imagesToProcess[imgIDB])
        #cv2.imwrite('C.jpg', imagesToProcess[imgIDC])
        #cv2.imwrite('D.jpg', imagesToProcess[imgIDD])
        #cv2.imwrite('0.jpg', imagesToProcess[0])
        cv2.waitKey()

        #imageToProcess = cv2.imread(args[0].image_path)
        #datum.cvInputData = imageToProcess
        #opWrapper.emplaceAndPop([datum])


        
        projectiveMatrix0 = np.dot(cameraIntrinsics[0],cameraExtrinsics[0])
        projectiveMatrix1 = np.dot(cameraIntrinsics[1],cameraExtrinsics[1])
        projectiveMatrix11 = np.dot(cameraIntrinsics[11],cameraExtrinsics[11])
        #pixels = [None] * (numberOfCameras-1)

        for i in range(numberOfCameras-1):
            j = i+1
            worldCoord = np.array([cameraFromWorld[j][0], cameraFromWorld[j][1], cameraFromWorld[j][2], 1])
            pixel = np.dot(projectiveMatrix0, worldCoord)
            pixel = pixel/pixel[2]
            print('camera: ' + str(j) + ' with pixel ' + str(pixel))
            color = (0,0,255)
            #pixels[i] = pixel
            cv2.circle(imagesToProcess[0], (int(pixel[0]),int(pixel[1])),3,color,2)


        for i in range(numberOfCameras):
            j = i
            if i == 1:
                continue
            worldCoord = np.array([cameraFromWorld[j][0], cameraFromWorld[j][1], cameraFromWorld[j][2], 1])
            pixel = np.dot(projectiveMatrix1, worldCoord)
            pixel = pixel/pixel[2]
            print('camera: ' + str(j) + ' with pixel ' + str(pixel))
            color = (0,0,255)
            #pixels[i] = pixel
            cv2.circle(imagesToProcess[1], (int(pixel[0]),int(pixel[1])),3,color,2)

        for i in range(numberOfCameras):
            j = i
            if i == 11:
                continue
            worldCoord = np.array([cameraFromWorld[j][0], cameraFromWorld[j][1], cameraFromWorld[j][2], 1])
            pixel = np.dot(projectiveMatrix11, worldCoord)
            pixel = pixel/pixel[2]
            print('camera: ' + str(j) + ' with pixel ' + str(pixel))
            color = (0,0,255)
            #pixels[i] = pixel
            cv2.circle(imagesToProcess[11], (int(pixel[0]),int(pixel[1])),3,color,2)

        #projectiveMatrix11 = np.dot(cameraIntrinsics[11],cameraExtrinsics[11])

        #worldCoord11 = np.array([cameraExtrinsics[10][0,3], cameraExtrinsics[10][1,3], cameraExtrinsics[10][2,3], 1])

        #pixel = np.dot(projectiveMatrix0, worldCoord11)
        #pixel = pixel/pixel[2]
        #print(pixel)
        
        
        cv2.imshow('0', imagesToProcess[0])
        cv2.imshow('1', imagesToProcess[1])
        cv2.imshow('11', imagesToProcess[11])

        cv2.imwrite('0.jpg', imagesToProcess[0])
        cv2.imwrite('1.jpg', imagesToProcess[1])
        cv2.imwrite('11.jpg', imagesToProcess[11])


        cv2.waitKey()
    #except Exception as e:
    #    print(e)
    #    sys.exit(-1)
