import numpy as np

# 张量类
class Tensor:
    def __init__(self, values, requires_grad=False, dependency=None):
        self._values = np.array(values, dtype=np.float32)
        self.shape = self._values.shape
        self.requires_grad = requires_grad
        self.grad = None
        self.dependency = dependency if dependency is not None else []

        if requires_grad:
            self.zero_grad()

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, new_values):
        self._values = np.array(new_values, dtype=np.float32)
        self.grad = None

    def zero_grad(self):
        self.grad = np.zeros(self.shape, dtype=np.float32)

    def backward(self, grad=None):
        assert not (grad is None and self._values.size > 1), "grad can be implicitly created only for scalar outputs"

        grad = 1.0 if grad is None else grad
        grad = np.array(grad, dtype=np.float32)

        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

        for dep in self.dependency:
            grad_for_dep = dep["grad_calculate_fn"](self.grad)
            dep["tensor"].backward(grad_for_dep)

# 辅助函数：将输入转换为 Tensor
def as_tensor(obj):
    if isinstance(obj, Tensor):
        return obj
    return Tensor(obj)

# 构建一元操作的计算图
def build_unary_op(input1, out, fn1):
    if input1.requires_grad:
        out.dependency.append(dict(tensor=input1, grad_calculate_fn=fn1))
    return out

# 构建二元操作的计算图
def build_binary_op(input1, input2, out, fn1, fn2):
    if input1.requires_grad:
        out.dependency.append(dict(tensor=input1, grad_calculate_fn=fn1))
    if input2.requires_grad:
        out.dependency.append(dict(tensor=input2, grad_calculate_fn=fn2))
    return out

# 操作基类
class OPS:
    def __init__(self, name=None, op_type=None):
        self.name = name
        self.op_type = op_type

    def __call__(self, *args):
        raise NotImplementedError

# 矩阵乘法操作
class MatMul(OPS):
    def __init__(self, name=None):
        super(MatMul, self).__init__(name=name, op_type="matmul")
        self.input1 = None
        self.input2 = None

    def __call__(self, input1, input2):
        self.input1 = as_tensor(input1)
        self.input2 = as_tensor(input2)

        out = Tensor(np.matmul(self.input1.values, self.input2.values),
                     requires_grad=(self.input1.requires_grad or self.input2.requires_grad))

        return build_binary_op(self.input1, self.input2, out,
                               self.matmul_grad_calculate_fn_1, self.matmul_grad_calculate_fn_2)

    def matmul_grad_calculate_fn_1(self, grad):
        return np.matmul(grad, self.input2.values.T)

    def matmul_grad_calculate_fn_2(self, grad):
        return np.matmul(self.input1.values.T, grad)

# ReLU 操作
class ReLU(OPS):
    def __init__(self, name=None):
        super(ReLU, self).__init__(name=name, op_type="relu")
        self.input1 = None

    def __call__(self, input1):
        self.input1 = as_tensor(input1)
        out = Tensor(np.maximum(0, self.input1.values),
                     requires_grad=self.input1.requires_grad)

        return build_unary_op(self.input1, out, self.relu_grad_calculate_fn)

    def relu_grad_calculate_fn(self, grad):
        return grad * (self.input1.values > 0).astype(np.float32)

# 测试代码
if __name__ == "__main__":
    # 测试矩阵乘法
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = Tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)

    matmul = MatMul()
    z = matmul(x, y)  # z = x @ y

    # 测试 ReLU
    relu = ReLU()
    out = relu(z)

    # 假设损失函数是 out 的和
    loss = out.values.sum()
    loss_tensor = Tensor(loss, requires_grad=True, dependency=[
        {"tensor": out, "grad_calculate_fn": lambda grad: np.ones_like(out.values) * grad}
    ])

    # 反向传播
    loss_tensor.backward()

    print("Input x:\n", x.values)
    print("Input y:\n", y.values)
    print("Output z (after matmul):\n", z.values)
    print("Output after ReLU:\n", out.values)
    print("Gradient of x:\n", x.grad)
    print("Gradient of y:\n", y.grad)