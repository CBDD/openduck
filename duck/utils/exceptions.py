class EquilibrationError(Exception):
    """Raised when a simulation log shows an unstable (non-equilibrated) system."""

    def __init__(self, msg="System is not equilibrated.", *args, **kwargs):
        super().__init__(msg, *args, **kwargs)
