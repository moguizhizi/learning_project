from typing import NamedTuple


class Person(NamedTuple):
    name: str
    age: int


person = Person(name="XiaoMing", age=18)
print(person.name)
print(person.age)