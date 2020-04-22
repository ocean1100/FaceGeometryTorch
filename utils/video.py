import cv2

def video_to_images(video_path, max_iter):
    cap = cv2.VideoCapture(video_path)
    i = 0
    images = []
    while (cap.isOpened() and i < max_iter):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            images.append(frame)
            i = i+1
        # Break the loop
        else: 
            break
    return images

def save_images_to_video(input_images_p, output_path):
    frame = cv2.imread(input_images_p[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter(output_path, 0, 30, (width,height))

    for img_p in input_images_p:
        video.write(cv2.imread(img_p))
    cv2.destroyAllWindows()
    video.release()