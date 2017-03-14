
class ModelNotTrainedException(Exception):
    """Exception raised for untrained models.
    Changelog:
    - 14/3 KS First commit
    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class InvalidDatasetException(Exception):
    """Exception raised for invalid datasets.
    Changelog:
    - 14/3 KS First commit
    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message