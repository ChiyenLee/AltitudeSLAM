import cv2
import matplotlib.pyplot as plt 
import numpy as np
import exif
import utm
from extract_gt import getGPSFromImage


class VisualOdometry:
    def __init__(self, vis=False):
        """
        Initialize class that extract visual odometry between two frames
        """
        self.orb = cv2.ORB_create(4000)
        index_params= dict(algorithm = 6, # FLANN_INDEX_LSH
                            table_number = 6, # 12
                            key_size = 12,     # 20
                            multi_probe_level = 2) #2
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        # self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.down_sample_ratio = 5
        self.vis = vis

    def estimateOdometry(self, image1, image2):
        """
        Estimate odometry between two images
        """
        image1 = cv2.resize(image1, (int(image1.shape[1]/self.down_sample_ratio), int(image1.shape[0]/self.down_sample_ratio)))
        image2 = cv2.resize(image2, (int(image2.shape[1]/self.down_sample_ratio), int(image2.shape[0]/self.down_sample_ratio)))

        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        kp1, descriptors1 = self.orb.detectAndCompute(gray1, None)
        kp2, descriptors2 = self.orb.detectAndCompute(gray2, None)

        matches = self.matcher.knnMatch(descriptors1, descriptors2, 2)
        src_pts = []
        dst_pts = []
        # Filter out the bad ones 
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8*n.distance:
                src_pts.append(kp1[m.queryIdx].pt)
                dst_pts.append(kp2[m.trainIdx].pt)
        src_pts = np.array(src_pts)
        dst_pts = np.array(dst_pts)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        src_pts_image = np.hstack((src_pts,np.ones((src_pts.shape[0],1))))
        target_pts_image = H @ src_pts_image.T
        target_pts_image = target_pts_image.T / target_pts_image.T[:,2,None]
        target_pts_image = target_pts_image[:,:2]

        K = self.getIntrinsics()
        mask,_ = np.where(mask)

        ret_val, R,t, normals = cv2.decomposeHomographyMat(H, K)
        src_pts_rected = cv2.undistortPoints(src_pts[mask].astype(np.float32), K, np.zeros(4,dtype=np.float32))
        dst_pts_rected = cv2.undistortPoints(dst_pts[mask].astype(np.float32), K, np.zeros(4, dtype=np.float32))
        possible_solutions = cv2.filterHomographyDecompByVisibleRefpoints(R,normals, src_pts_rected, dst_pts_rected)
        breakpoint()
        # matches = self.matcher.match(descriptors1, descriptors2)
        if self.vis:
            # Need to draw only good matches, so create a mask
            # matchesMask = [[0,0] for i in range(len(matches))]

            # # ratio test as per Lowe's paper
            # for i,(m,n) in enumerate(matches):
            #     if m.distance < 0.6*n.distance:
            #         matchesMask[i]=[1,0]

            
            # draw_params = dict(matchColor = (0,255,0),
            #                 singlePointColor = (255,0,0),
            #                 matchesMask = matchesMask,
            #                 flags = 0)

            # final_img = cv2.drawMatchesKnn(image1, kp1, image2, kp2, matches, None, **draw_params)
            # # matches = sorted(matches, key=lambda x: x.distance)
            # # final_img = cv2.drawMatches(image1, kp1, image2, kp2, matches[:50], None)
            # final_img = cv2.resize(final_img, ( 1330, 500))
            
            # plt.imshow(final_img)
            plt.figure(1)
            plt.subplot(1,2,1)
            plt.plot(dst_pts[mask,0], dst_pts[mask,1], ".", label="dst pts")
            plt.plot(target_pts_image[mask,0], target_pts_image[mask,1], ".", label="dst pts transformed")
            plt.legend()
            plt.imshow(image2)

            plt.subplot(1,2,2)
            plt.imshow(image1)
            plt.plot(src_pts[mask,0], src_pts[mask,1], ".")
            plt.show()
        return R, t, normals



    def getIntrinsics(self):
        """
        A rough ballpark of the sensor intrinsics of parrot sequoia        
        https://www.pix4d.com/product/sequoia/faq#6-Detailed-Sensor-Specifications
        """
        pixel_size = 1.34 # us 
        focal_length = 4.88 # ms 
        image_width = 4608 
        image_height = 3456 
        fx = (focal_length * 1e3) / pixel_size
        fy = fx
        cx = int((image_width-1)/ 2)
        cy = int((image_height-1)/ 2)

        A = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0,  1]])

        return A/self.down_sample_ratio


if __name__ == "__main__":
    VO = VisualOdometry(vis=False)
    path1 = "geotagged-images/IMG_1013.JPG"
    path2 = "geotagged-images/IMG_1014.JPG"
    image1 = cv2.imread(path1)
    image2 = cv2.imread(path2)

    lat1, long1 = getGPSFromImage(path1)
    lat2, long2 = getGPSFromImage(path2)
    u1 = utm.from_latlon(lat1, long1)
    u2 = utm.from_latlon(lat2, long2)
    u1 = np.array(u1[:2])
    u2 = np.array(u2[:2])
    t_true = u2 - u1

    
    print("ground truth: ", t_true)
    rotations, translations, normals = VO.estimateOdometry(image1, image2)
    breakpoint()
    for i in range(len(translations)):
        t = translations[i]
        scale = t_true[0] / t[0]
        print("estimated :", t * scale)
