import cv2

def get_coords(img_path):
    img = cv2.imread(img_path)
    coords = []
    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"{img_path}: clicked x={x} y={y}")
            coords.append((x,y))
            if len(coords) == 2:
                cv2.destroyAllWindows()
    cv2.namedWindow(img_path, cv2.WINDOW_NORMAL)
    cv2.imshow(img_path, img)
    cv2.setMouseCallback(img_path, click)
    cv2.waitKey(0)
    return coords

print("Click the left edge of the left character's body in seg_frame0.png, then click the same spot again to confirm")
get_coords('seg_frame0.png')

print("Now click the same spot in full_frame3540.png")
get_coords('full_frame3540.png')