import cv2
import pygame
import numpy as np

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture('animation.mp4')

pygame.mixer.init()
my_sound = pygame.mixer.Sound('hhhfff.mp3')

while 1:
    ret1, img1 = cap1.read()  # kamera
    ret2, frame2 = cap2.read()  # meine animation mp4 file

    if ret1:
        frame1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            frame1,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            # flags = cv2.CV_HAAR_SCALE_IMAGE
        )
        # print("Found {0} faces!".format(len(faces)))
        if len(faces) > 0:
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            frame1_tmp = img1[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1, :]
            frame1 = frame1[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1]
            eyes = eyeCascade.detectMultiScale(
                frame1,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                # flags = cv2.CV_HAAR_SCALE_IMAGE
            )  # “Closed-Eye-Detection-with-opencv" aus github gekopiert
            # https://github.com/GangYuanFan/Closed-Eye-Detection-with-opencv/blob/master/cv_close_eye_detect.py


            if len(eyes) == 0:
                # frame2_tmp = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                cv2.imshow('animation.mp4', frame2)
                waitkey = cv2.waitKey(1)
                print('augen sind zu!!!')
                my_sound.set_volume(0.0)
                # es fehlt noch: wenn man absichtlich lang zeit augen zumacht, loop play the video, (nicht notwendig)
                # es fehlt noch: if (cap.isOpened()== False): print("Es gibt Fehler, error opening file") (nicht notwendig)
            else:
                frame2_tmp = cv2.normalize(frame2, None, alpha=0, beta=255)  # ich mache meine animation ganz dunkel,
                cv2.imshow('animation.mp4', frame2_tmp)
                waitkey = cv2.waitKey(1)
                print('augen sind auf!!!')
                my_sound.play()
                my_sound.set_volume(1.0)

        else:
                # hsv is better to recognize color, convert the BGR frame to HSV
                hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
                # in hsv red color located in two region. Create the mask for red color
                # mask the red color and get an grayscale output where red is white
                # everything else are black
                mask = cv2.inRange(hsv, (0, 0, 0), (255, 255, 255))
                # get the index of the white areas and make them orange in the main frame
                for i in zip(*np.where(mask == 255)):
                    frame2[i[0], i[1], 0] = 202
                    frame2[i[0], i[1], 1] = 95
                    frame2[i[0], i[1], 2] = 36

                cv2.namedWindow('animation.mp4', cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty('animation.mp4', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                # play the new video
                cv2.imshow('animation.mp4', frame2)
                # cv2.imshow('animation.mp4', frame2)
                waitkey = cv2.waitKey(1)
                print('no face!!!')
                my_sound.set_volume(0.0)

        if waitkey == ord('w') or waitkey == ord('W'):
            cap2.release()
            cv2.destroyAllWindows()
            break