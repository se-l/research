class EnumStr(str):
    def __str__(self):
        return self.name

    def lower(self):
        return self.name.lower()

    def upper(self):
        return self.name.upper()
