from abc import ABC, abstractmethod


class HandlerPattern(ABC):
    """
    Abstract base class for handler patterns.

    This class provides a template for creating handler patterns that can be
    used to process and validate models in a consistent manner.

    Methods:
        _check(model):
            Abstract method that must be implemented by subclasses.
            This method is responsible for checking or validating the provided model.

    """

    @abstractmethod
    def _check(self, model):
        """
        Check or validate the provided model.

        This abstract method must be implemented by any subclass. It should
        include the logic necessary to check or validate the model to ensure
        it meets the required criteria or standards.

        Args:
            model: The model to be checked or validated. The type and structure
                   of the model depend on the specific implementation in the subclass.

        """
        raise NotImplementedError("Not implemented '_check()'!")
