import streamlit as st
import cv2
import numpy as np
from PIL import Image

def get_frame_from_video(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_number >= total_frames:
        st.error(f"Frame number exceeds total frames in the video. The video has {total_frames} frames.")
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        st.error("Failed to read the frame from the video.")
        return None

    return frame

def motion_blur_kernel(length, angle):
    M = cv2.getRotationMatrix2D((length / 2, length / 2), angle, 1)
    psf = np.diag(np.ones(length))
    psf = cv2.warpAffine(psf, M, (length, length))
    psf = psf / psf.sum()
    return psf

def wiener_deconvolution(img, kernel, K):
    kernel = np.flipud(np.fliplr(kernel))
    dummy = np.copy(img)
    dummy = np.fft.fft2(dummy)
    kernel = np.fft.fft2(kernel, s=img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.fft.ifft2(dummy)
    dummy = np.abs(dummy)

    # Normalize the result to be in the range [0, 255]
    dummy = np.clip(dummy, 0, 255)
    return dummy.astype(np.uint8)

def deblur_color_image(img, kernel, K):
    # Split the image into its color channels
    b, g, r = cv2.split(img)
    
    # Apply Wiener deconvolution to each channel
    b_deblurred = wiener_deconvolution(b, kernel, K)
    g_deblurred = wiener_deconvolution(g, kernel, K)
    r_deblurred = wiener_deconvolution(r, kernel, K)
    
    # Merge the deblurred channels back together
    deblurred_img = cv2.merge((b_deblurred, g_deblurred, r_deblurred))
    
    return deblurred_img

def sharpen_image(image):
    # Create a kernel for sharpening
    sharpening_kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
    # Apply the sharpening kernel to the image
    sharpened = cv2.filter2D(image, -1, sharpening_kernel)
    
    return sharpened


def preprocessing(gray):
    sharpened_img = sharpen_image(gray)
    # sharpened_img2 = sharpen_image(sharpened_img)
    med_blurred = cv2.medianBlur(sharpened_img, 7)
    # med_blurred2 = cv2.medianBlur(med_blurred,7)
    # sharpened_img3 = sharpen_image(med_blurred)
    # plt.imshow(sharpened_img3)

    return med_blurred


def contour_drawer(gray_image,rgb_image, num, show_img = True, binary_thresh =(30,60), color = (0, 255, 0)):
    # rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # gray_image = cv2.GaussianBlur(gray_image, (5, 5), 1)
    lower_threshold = binary_thresh[0]
    upper_threshold = binary_thresh[1]

# Apply custom thresholding
    binary_image = np.zeros_like(gray_image)
    binary_image[(gray_image >= lower_threshold) & (gray_image <= upper_threshold)] = 255
    # _, binary_image = cv2.threshold(gray_image, binary_thresh[0], binary_thresh[1], cv2.THRESH_BINARY)
    img_copy = rgb_image.copy()
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    if show_img == True:	
        img_copy2 = rgb_image.copy()
        
        img_copy2 = cv2.drawContours(img_copy2, contours[num], -1, color, 10)
        # plt.imshow(img_copy2)


    return contours, len(contours), binary_image   




def contour_approximation(image, contours,num,precision):
    epsilon = precision * cv2.arcLength(contours[0], True)

    # Approximate the contour
    approx = cv2.approxPolyDP(contours[num], epsilon, True)
    # smooth = rgb_image.copy()
    # Draw the approximated contour (in red)
    # smooth = cv2.drawContours(smooth, [approx], -1, (0, 255, 0), 20)
    # plt.imshow(smooth)

    return 0,approx





def fit_circle(points):
    x = points[:, 0]
    y = points[:, 1]
    A = np.column_stack([x, y, np.ones(x.shape[0])])
    b = x**2 + y**2
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    center_x = c[0] / 2
    center_y = c[1] / 2
    radius = np.sqrt(c[2] + center_x**2 + center_y**2)
    return center_x, center_y, radius


def convert_mm(dis):
    return (0.0406969697*dis)


def measure_euclidean(p1,p2):
    # print(p1)
    x1, y1 = p1
    x2, y2 = p2
    
    dist = ((x2-x1)**2 + (y2-y1)**2)**0.5
    return dist



def main():
    st.title("Mearements in mm")
    
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    
    length = 7  # Length of the motion blur
    angle = 180   # Angle of the motion blur
    K = 0.007     # Regularization constant for Wiener filter

    # Generate kernel
    kernel = motion_blur_kernel(length, angle)

    # Deblur the color image
    # plt.imshow(deblurred_img)


    if uploaded_file is not None:
        video_path = uploaded_file.name
        with open(video_path, mode='wb') as f:
            f.write(uploaded_file.read())  # Save the uploaded file to disk
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        st.write(f"The video has {total_frames} frames.")

        frame_number = st.number_input("Enter the frame number you want to extract", min_value=0, max_value=total_frames-1, step=1, value=0)
        
        if st.button("Extract Frame"):
            frame = get_frame_from_video(video_path, frame_number)
            if frame is not None:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame, channels="BGR", caption=f"Frame {frame_number}")
                deblurred_img = deblur_color_image(rgb_image, kernel, K)
                


                gray = cv2.cvtColor(deblurred_img, cv2.COLOR_RGB2GRAY)

                final_image = preprocessing(gray)
                contours, length,bin_image = contour_drawer(final_image,rgb_image, 0,binary_thresh=(130,250))
                # print(length)

                smoothened_image, approx = contour_approximation(final_image, contours,0,0.03)
                image_c = bin_image.copy()
                image_c = cv2.cvtColor(image_c, cv2.COLOR_GRAY2BGR)
# plt.imshow(image_c)
                x = [i[0][0] for i in approx]
                y = [i[0][1] for i in approx]
                for i in approx:
                    image_c = cv2.circle(image_c, i[0], 15, (0,0,255), -1)

                # st.image(image_c, channels="BGR", caption=f"Frame {frame_number}")

                contours, length, bin_image = contour_drawer(final_image,rgb_image, 5, binary_thresh=(130,250))
                circle = np.array(contours[5])

                circ_contour = contours[5]
                circ_contour.shape = (circ_contour.shape[0], 2)

                center_x, center_y, radius = fit_circle(circ_contour)
               
                cv2.circle(image_c, (int(center_x), int(center_y)), int(radius), (0, 255, 0), 15)
                cv2.circle(image_c, (int(center_x), int(center_y)), 15, (0,0,255), -1)

                st.image(image_c, channels="BGR", caption=f"Frame {frame_number}")

                st.write("## Distances between points:")
                distances = [
                    (measure_euclidean(approx[0][0], approx[3][0]), convert_mm(measure_euclidean(approx[0][0], approx[3][0]))),
                    (measure_euclidean(approx[0][0], approx[1][0]), convert_mm(measure_euclidean(approx[0][0], approx[1][0]))),
                    (measure_euclidean(approx[1][0], approx[2][0]), convert_mm(measure_euclidean(approx[1][0], approx[2][0]))),
                    (measure_euclidean(approx[2][0], approx[3][0]), convert_mm(measure_euclidean(approx[2][0], approx[3][0])))
                ]

                for i, (dist, dist_mm) in enumerate(distances):
                    st.write(f"Distance {i+1}: {dist:.2f} pixels  -->  {dist_mm:.2f} mm")

                st.write("## Distances from points to circle center:")
                circle_distances = [
                    (measure_euclidean(approx[0][0], [int(center_x), int(center_y)]), convert_mm(measure_euclidean(approx[0][0], [int(center_x), int(center_y)]))),
                    (measure_euclidean(approx[3][0], [int(center_x), int(center_y)]), convert_mm(measure_euclidean(approx[3][0], [int(center_x), int(center_y)]))),
                    (measure_euclidean(approx[1][0], [int(center_x), int(center_y)]), convert_mm(measure_euclidean(approx[1][0], [int(center_x), int(center_y)]))),
                    (measure_euclidean(approx[2][0], [int(center_x), int(center_y)]), convert_mm(measure_euclidean(approx[2][0], [int(center_x), int(center_y)])))
                ]

                for i, (dist, dist_mm) in enumerate(circle_distances):
                    st.write(f"Distance from Point {i+1} to Circle Center: {dist:.2f} pixels  -->  {dist_mm:.2f} mm")
                
if __name__ == "__main__":
    main()
