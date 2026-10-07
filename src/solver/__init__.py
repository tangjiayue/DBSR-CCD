from typing import Dict

from ._solver import BaseSolver
from .clas_solver import ClasSolver
from .det_solver import DetSolver

TASKS: Dict[str, BaseSolver] = {
    "classification": ClasSolver,
    "detection": DetSolver,
}
