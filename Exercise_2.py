class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary

    def get_details(self):
        print(f"The employee's name is {self.name} and his/her salary is {self.salary}€")

class Manager(Employee):
    def promote(self, employee):
        employee.salary += (employee.salary * 10) / 100.0

class Developer(Employee):
    def __init__(self, name, salary, programming_language):
        super().__init__(name, salary)
        self.programming_language = programming_language

manager = Manager(name='Stavros',salary=1600)
developer = Developer(name='Anastasia',salary=1000, programming_language='Python')
manager.promote(developer)
manager.get_details()
developer.get_details()