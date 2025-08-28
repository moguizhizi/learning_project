from pydantic import BaseModel, conint
from typing import Optional


class User(BaseModel):
    name: str
    age: conint(ge=0, le=120)
    address: str
    def aa(self, ):
        print("5285")


user = User(name="1", age=30, address="52")
print(user)

user = User(name="1", age="30", address="52")
print(user)


user = User(name="1", address="52", age=50)
print(user)
user.aa()

