# def my_decorator(func):
#     def wrapper(*args, **kwargs):
#         """This function says wrapper"""
#         print("Before function call")
        
#         print(args)
#         print(kwargs)
        
#         return func(*args, **kwargs)
#     return wrapper

# @my_decorator
# def say_hello():
#     """This function says hello"""
#     print("Hello")

# print(say_hello.__name__)  # 输出: wrapper
# print(say_hello.__doc__)   # 输出: None
# print(say_hello())


from functools import wraps

def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("Before function call")
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def say_hello():
    """This function says hello"""
    print("Hello")

print(say_hello.__name__)  # 输出: say_hello
print(say_hello.__doc__)   # 输出: This function says hello
say_hello()
