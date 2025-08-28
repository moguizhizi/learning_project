from enum import Enum

class Color(Enum):
    RED="red"
    BLUE="blue"
    YELLOW="yellow"

def print_enum(color:Color):
    print(color)
    
print_enum(Color.RED)
print_enum(Color.BLUE)
print_enum("d")

def set_color(color: Color) -> None:
    print(f"Setting color to {color.name}")

set_color(Color.RED)    # 合法
set_color("RED")        # 非法，mypy 报错