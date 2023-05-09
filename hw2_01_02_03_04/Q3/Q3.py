import cv2
from tqdm import tqdm
import numpy as np

class Question3:
    def __init__(self) -> None:
        pass

    def perspectiveTransform(self, videoPath, imagePath):
        if videoPath == None or imagePath == None:
            print("please select video and image")
            return
        
        frames = []
        output_path = videoPath[:-4] + "_output.mp4"

        capture = cv2.VideoCapture(videoPath)
        img = cv2.imread(imagePath)
        FPS = capture.get(cv2.CAP_PROP_FPS)

        # get all frames
        while capture.isOpened():
            ret, frame = capture.read()

            # if frame is read correctly ret is True
            if not ret:
                print("All frames saved in variable \"frames\"")
                break
            
            frames.append(frame)

            
        # video writer
        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), FPS, (round(
            2 * capture.get(cv2.CAP_PROP_FRAME_WIDTH)), round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        print("Video processing...")
        for frame in tqdm(frames):
            arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            arucoParams = cv2.aruco.DetectorParameters_create()            
            markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

            if markerIds.shape[0] != 4:
                continue

            # upper left corner
            index = np.squeeze(np.where(markerIds == 1))
            refPt1 = np.squeeze(markerCorners[index[0]])[0] 
            
            # upper right corner
            index = np.squeeze(np.where(markerIds == 2))
            refPt2 = np.squeeze(markerCorners[index[0]])[1]
            
            distance = np.linalg.norm(refPt1 - refPt2)

            scalingFac = 0.005

            pts_dst = [[refPt1[0] - round(scalingFac * distance), refPt1[1] - round(scalingFac * distance)]]

            pts_dst = pts_dst + [[refPt2[0] + round(scalingFac * distance), refPt2[1] - round(scalingFac * distance)]]

            # lower right corner
            index = np.squeeze(np.where(markerIds == 3))
            refPt3 = np.squeeze(markerCorners[index[0]])[2]
            
            pts_dst = pts_dst + [[refPt3[0] + round(scalingFac * distance), refPt3[1] + round(scalingFac * distance)]]

            # lower left corner
            index = np.squeeze(np.where(markerIds == 4))
            refPt4 = np.squeeze(markerCorners[index[0]])[3]
            
            pts_dst = pts_dst + [[refPt4[0] - round(scalingFac * distance), refPt4[1] + round(scalingFac * distance)]]

            pts_src = [[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]]

            pts_src_m = np.asarray(pts_src)
            pts_dst_m = np.asarray(pts_dst)

            # Calculate Homography
            h, status = cv2.findHomography(pts_src_m, pts_dst_m)

            # Warp source image to destination based on homography
            warped_image = cv2.warpPerspective(img, h, (frame.shape[1], frame.shape[0]))

            # Prepare a mask representing region to copy from the warped image into the original frame.
            mask = np.zeros([frame.shape[0], frame.shape[1]], dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32([pts_dst_m]), (255, 255, 255), cv2.LINE_AA)

            # Erode the mask to not copy the boundary effects from the warping
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.erode(mask, element, iterations=3)

            # Copy the mask into 3 channels.
            warped_image = warped_image.astype(float)
            mask3 = np.zeros_like(warped_image)
            for i in range(3):
                mask3[:, :, i] = mask / 255

            # Copy the warped image into the original frame in the mask region.
            warped_image_masked = cv2.multiply(warped_image, mask3)
            frame_masked = cv2.multiply(frame.astype(float), 1 - mask3)
            im_out = cv2.add(warped_image_masked, frame_masked)

            # Showing the original image and the new output image side by side
            concatenatedOutput = cv2.hconcat([frame.astype(float), im_out])

            # write frames to the file
            video_writer.write(concatenatedOutput.astype(np.uint8))

        capture.release()
        video_writer.release()
        print("Transfrom done...")

        # play the video
        self.playVideo(output_path=output_path, fps=FPS)

    def playVideo(self, output_path, fps):
        capture_out = cv2.VideoCapture(output_path)


        while capture_out.isOpened():
            ret, frame = capture_out.read()

            # if frame is read correctly ret is True
            if not ret:
                print("Video played")
                break
            
            # cv2.imshow("Perspective Transform", cv2.resize(frame, dsize=None, fx=0.37, fy=0.37))    # resize to show completelly
            cv2.imshow("Perspective Transform", cv2.resize(frame, dsize=None, fx=0.7, fy=0.7))  # resize to show completelly
            # press "q" to exit playing
            if cv2.waitKey(round(fps)) == ord('q'):
                break
        
        capture_out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # for testing
    Q3 = Question3()
    Q3.perspectiveTransform("C:\\Users\\Xi\\Desktop\\cvdl_hw2\\Dataset_CvDl_Hw2\\Q3_Image\\video.mp4", "C:\\Users\\Xi\\Desktop\\cvdl_hw2\\Dataset_CvDl_Hw2\\Q3_Image\\logo.png")
    Q3.playVideo("C:\\Users\\Xi\\Desktop\\cvdl_hw2\\Dataset_CvDl_Hw2\\Q3_Image\\video_output.mp4", 30)