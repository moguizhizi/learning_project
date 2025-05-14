class MyClass:
    def old_func(self):
        return "Old"

obj = MyClass()
def new_func():
    return "New"

# 使用 setattr 替换方法
setattr(obj, "old_func", new_func)
print(obj.old_func())  # 输出: New