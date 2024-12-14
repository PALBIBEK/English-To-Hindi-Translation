class Evidence:
    def __init__(self,id=None,context: str = None,title : str = None):
        self.id=id
        self.text=context
        self.title=title

    def id(self):
        return self.id

    def text(self):
        return self.context
    
    def title(self):
        return self.title