import cv2

# 图像路径
p = "learning_project/images/phone/Click_on_ticket_settlement.jpg"

# 矩形参数
rect = {'x': 135, 'y': 430, 'width': 5, 'height': 5}

# 读取图像
image = cv2.imread(p)

if image is not None:
    # 计算矩形坐标
    x, y = rect['x'], rect['y']
    width, height = rect['width'], rect['height']

    # 左上角和右下角坐标
    start_point = (x, y)
    end_point = (x + width, y + height)

    # 绘制矩形 (颜色为绿色，线宽为2)
    color = (0, 255, 0)  # BGR格式
    print(start_point)
    print(end_point)
    thickness = 2
    cv2.rectangle(image, start_point, end_point, color, thickness)

    # 保存修改后的图像
    output_path = "learning_project/output/Click_on_ticket_settlement_highlight.jpg"  # 保存路径
    cv2.imwrite(output_path, image)
    print(f"图像已保存到 {output_path}")

    # 显示结果
    # cv2.imshow('Image with Rectangle', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
else:
    print("无法加载图像，请检查路径是否正确")