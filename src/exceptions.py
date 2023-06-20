"""
Class containing custom errors for the simulator
"""

class ShapeError(Exception):
    """
    Exception raised for wrong size passed into hat map
    """
    def __init__(self):
        super().__init__("Shape of input vector incorrect")