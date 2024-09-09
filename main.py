import tkinter as tk
from tkinter import filedialog as tkfiledialog
import cv2
from PIL import Image
from PIL import ImageTk
from tkinter import Label 
import numpy as np
import matplotlib.pyplot as plt

list = []

def select_image():
    global frame1 
    if len(list) != 0:
        input_delete()
        
    path = tkfiledialog.askopenfilename()
    list.append(path)
    
    if len(path) > 0:
        image_input = cv2.imread(path)
        image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        # Convert the images to PIL format
        image_pil = Image.fromarray(image_rgb)
        # Convert the PIL format to ImageTk format
        image = ImageTk.PhotoImage(image_pil)
        frame1 = Label(image=image)
        frame1.image = image 
        frame1.pack(side="left", fill=tk.BOTH, expand=True)

def input_delete():
    list.clear()
    frame1.destroy()

def get_threshold(entry):
    try:
        value = int(entry.get())
        if 0 <= value <= 255:
            return value
        else:
            raise ValueError
    except ValueError:
        return None

def line_detection():
    min_threshold = get_threshold(entry1)
    max_threshold = get_threshold(entry2)
    
    if min_threshold is None | max_threshold is None:
        print("Invalid threshold values. Please enter integers between 0 and 255.")
        return
    
    image = cv2.imread(list[0])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, min_threshold, max_threshold, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    
    if lines is not None:
        for r_theta in lines:
            arr = np.array(r_theta[0], dtype=np.float64)
            r, theta = arr
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * r
            y0 = b * r
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    cv2.imshow("Hough Lines", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    list.clear()
    frame1.destroy()

def circle_detection():
    min_threshold = get_threshold(entry1)
    max_threshold = get_threshold(entry2)
    
    if min_threshold is None | max_threshold is None:
        print("Invalid threshold values. Please enter integers between 0 and 255.")
        return
    
    image = cv2.imread(list[0])
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(gray_img, 5)
    circles = cv2.HoughCircles(
        img_blur, cv2.HOUGH_GRADIENT, 1, 120,
        param1=max_threshold, param2=min_threshold,
        minRadius=int(entry3.get()), maxRadius=0
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 3)
            cv2.circle(image, (i[0], i[1]), 1, (0, 0, 255), 3)
    
    cv2.imshow("Hough Circles", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    list.clear()
    frame1.destroy()

def edge_detection():
    image = cv2.imread(list[0], cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 100, 200)
    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    list.clear()
    frame1.destroy()

window = tk.Tk()
window.geometry("900x800")
window.title("Hough Transform")
frame1 = None
frame2 = None

# Menu for manipulating
frame3 = tk.LabelFrame(window, bd=2, relief="ridge", text="Menu")
frame3.pack(fill="x")
lbl1 = tk.Label(frame3, text="Threshold(Min):")
lbl1.pack(side=tk.LEFT)
entry1 = tk.Entry(frame3, width=5)
entry1.pack(side="left")
lbl2 = tk.Label(frame3, text="Threshold(Max):")
lbl2.pack(side=tk.LEFT)
entry2 = tk.Entry(frame3, width=5)
entry2.pack(side="left")
btn1 = tk.Button(frame3, text="Straight-line", command=line_detection)
btn1.pack(side="left")
lbl3 = tk.Label(frame3, text="Search radius:")
lbl3.pack(side=tk.LEFT)
entry3 = tk.Entry(frame3, width=5)
entry3.pack(side="left")
btn2 = tk.Button(frame3, text="Circle", command=circle_detection)
btn2.pack(side="left")
btn3 = tk.Button(frame3, text="Edge", command=edge_detection)
btn3.pack(side="left")
btn4 = tk.Button(window, text="Select an image", command=select_image)
btn4.pack(side="bottom", fill="both", expand="yes", padx="5", pady="5")
window.mainloop()
