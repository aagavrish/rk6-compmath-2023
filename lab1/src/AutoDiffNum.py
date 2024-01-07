class AutoDiffNum:
    def __init__(self, a, b):
        self.real = a
        self.dual = b

    def __repr__(self):
        if self.dual < 0:
            return f'{self.real} - {-self.dual}e'
        return f'{self.real} + {self.dual}e'

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return AutoDiffNum(self.real + other, self.dual)
        return AutoDiffNum(self.real + other.real, self.dual + other.dual)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return AutoDiffNum(self.real - other, self.dual)
        return AutoDiffNum(self.real - other.real, self.dual - other.dual)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return AutoDiffNum(self.real * other, self.dual * other)
        return AutoDiffNum(self.real * other.real, self.dual * other.real + self.real * other.dual)

    def __pow__(self, power):
        return AutoDiffNum(self.real ** power, power * self.real ** (power - 1) * self.dual)
