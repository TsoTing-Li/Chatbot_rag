class DirectoryCheckError(Exception):
        """Base class for exceptions in this module."""
        pass

class MultipleJsonFilesError(DirectoryCheckError):
    """Exception raised for having multiple JSON files."""
    pass

class NoJsonFilesError(DirectoryCheckError):
    """Exception raised for not having any JSON files."""
    pass

class MissingKeysError(DirectoryCheckError):
    """Exception raised for missing keys in the JSON file."""
    pass

class ImageFolderNotFoundError(DirectoryCheckError):
    """Exception raised for not finding an image folder."""
    pass

class ImageNotFoundError(DirectoryCheckError):
    """Exception raised for missing images specified in the JSON file."""
    pass