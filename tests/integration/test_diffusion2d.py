"""
Tests for functionality checks in class SolveDiffusion2D
"""
import numpy as np
import pytest
from diffusion2d import SolveDiffusion2D

@pytest.fixture
def initial_values():
    return {
        "w": 5.0, "h": 5.0, "dx": 0.2, "dy": 0.2, "d": 3.5, "T_cold": 290.0, "T_hot": 750.0}

def test_initialize_physical_parameters(initial_values):
    """
    Checks function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()
    solver.initialize_domain(w=initial_values["w"], h=initial_values["h"], dx=initial_values["dx"], dy=initial_values["dy"])

    d = initial_values["d"]

    dx2, dy2 = solver.dx * solver.dx, solver.dy * solver.dy
    dt = dx2 * dy2 / (2 * d * (dx2 + dy2))

    solver.initialize_physical_parameters(d=initial_values["d"], T_cold=initial_values["T_cold"], T_hot=initial_values["T_hot"])
    assert solver.dt == dt

def test_set_initial_condition(initial_values):
    """
    Checks function SolveDiffusion2D.get_initial_function
    """
    solver = SolveDiffusion2D()
    solver.initialize_domain(w=initial_values["w"], h=initial_values["h"], dx=initial_values["dx"], dy=initial_values["dy"])
    solver.initialize_physical_parameters(d=initial_values["d"], T_cold=initial_values["T_cold"], T_hot=initial_values["T_hot"])    
    u_test = solver.T_cold * np.ones((solver.nx, solver.ny))
    r, cx, cy = 2, 5, 5
    r2 = r ** 2
    for i in range(solver.nx):
        for j in range(solver.ny):
            p2 = (i * solver.dx - cx) ** 2 + (j * solver.dy - cy) ** 2
            if p2 < r2:
                u_test[i, j] = solver.T_hot

    ground_truth = u_test.copy()

    test_value = solver.set_initial_condition()

    assert np.array_equal(ground_truth, test_value)