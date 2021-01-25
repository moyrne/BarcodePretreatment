import cv2
import imutils
import numpy as np


def main():
    # 读取文件
    gray = cv2.imread("image/IMG_20210119_191535.jpg", cv2.IMREAD_GRAYSCALE)
    cv2.imshow("gray", gray)
    grad_x = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad_y = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    # 用x方向的梯度减去y方向的梯度
    gradient = cv2.subtract(grad_x, grad_y)
    gradient = cv2.convertScaleAbs(gradient)
    # cv2.imshow("gradient", gradient)

    blurred = cv2.blur(gradient, (7, 7))
    (_, thresh) = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
    # cv2.imshow("thresh", thresh)

    # 　构造一个闭合核并应用于阈值图片
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    cv2.imshow("closed", closed)

    # 执行一系列的腐蚀和膨胀操作
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)
    cv2.imshow("closed2", closed)

    rst = cv2.findChessboardCorners(closed, (7, 7))
    print(rst)

    # 找到阈值化后图片中的轮廓，然后进行根据区域进行排序，仅保留最大区域
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    # 计算最大轮廓的旋转边界框
    rect = cv2.minAreaRect(c)
    print(rect)
    box = np.int0(cv2.boxPoints(rect))
    print(box)
    # 在检测到的条形码周围绘制边界框并显示图片
    image = cv2.imread("image/IMG_20210119_191535.jpg")
    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    cv2.imshow("Image", image)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
