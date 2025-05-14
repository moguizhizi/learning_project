import cv2

# 图像路径
p = "learning_project/images/phone/Turn_on_the_Dark_theme_by_tapping_the_toggle_switch.jpg"

# 矩形参数（现在用于圆点，width 和 height 表示直径）
rect = {'x': 648, 'y': 1214, 'width': 20, 'height': 20}

# 读取图像
image = cv2.imread(p)

if image is not None:
    # 计算圆点坐标和半径
    x, y = rect['x'], rect['y']
    radius = min(rect['width'], rect['height']) // 2  # 半径取宽度或高度的最小值的一半

    # 绘制实心圆点 (颜色为绿色，填充)
    color = (0, 255, 0)  # BGR格式
    print(f"Center: ({x}, {y}), Radius: {radius}")
    thickness = -1  # -1 表示填充
    cv2.circle(image, (x, y), radius, color, thickness)

    # 保存修改后的图像
    output_path = "learning_project/output/Turn_on_the_Dark_theme_by_tapping_the_toggle_switch_highlight.jpg"
    cv2.imwrite(output_path, image)
    print(f"图像已保存到 {output_path}")

    # 显示结果（可选，已注释）
    # cv2.imshow('Image with Circle', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
else:
    print("无法加载图像，请检查路径是否正确")