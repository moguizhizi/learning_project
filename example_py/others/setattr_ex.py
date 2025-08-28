class MyClass:
    def old_func(self):
        return "Old"

obj = MyClass()

def new_func():
    return "New"

# 使用 setattr 替换方法
setattr(obj, "old_func", new_func)
print(obj.old_func())  # 输出: New

def func_a():
    return f"A from"

def func_b():
    return f"B from"

class DynamicModule:
    def __init__(self, name, func_map):
        self.name = name
        for fname, f in func_map.items():
            setattr(self, fname, f)

funcs = {"do_a": func_a, "do_b": func_b}
mod = DynamicModule("Bob", funcs)

print(mod.do_a())  # 输出：A from Bob
print(mod.do_b())  # 输出：B from Bob
