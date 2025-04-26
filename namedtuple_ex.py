from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])

p = Point(x=4, y=5)
print(p.x)
print(p.y)

print(p[0])
print(p[1])

print(p._asdict())

Student = namedtuple('Student', 'name age grade')

student1 = Student(name="A", age=14, grade="65")
student2 = Student(name="A", age=14, grade="65")

print(student1)
print(student2)

print(student1==student2)

name, age, grade = student1
print(name)
print(age)
print(grade)

default_student = Student(name="unknown", age=14, grade="unkown")
student1 = default_student._replace(name=56)
print(student1)

Student = namedtuple('Student', 'name age grade', defaults=[19, "unkown", "unkown"])
student1 = Student(name="A")
print(student1)

Book = namedtuple('Book', 'title author year')

# 创建一个书籍列表
books_data = [
    ('The Great Gatsby', 'F. Scott Fitzgerald', 1925),
    ('1984', 'George Orwell', 1949),
    ('To Kill a Mockingbird', 'Harper Lee', 1960)
]

books = [Book(*book) for book in books_data]
for book in books:
    print(book)
    
Address = namedtuple("Address", 'street city country')
Person = namedtuple('Person', 'name age address')

address = Address(street="A", city="B", country="C")
person = Person(name="B", age=85, address=address)
print(person)

