"""
Tests for functions in class SolveDiffusion2D
"""

import pytest, numpy as np
from diffusion2d import SolveDiffusion2D


def test_initialize_domain():
    """
    Check function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()
    solver.initialize_domain(w=5.0, h=5.0, dx=0.2, dy=0.2)
    assert solver.nx == 25
    assert solver.ny == 25


def test_initialize_physical_parameters():
    """
    Checks function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()
    solver.dx = 0.2
    solver.dy = 0.2
    solver.D = 3.5
    solver.initialize_physical_parameters(d=3.5, T_cold=290.0, T_hot=750.0)
    assert solver.dt == pytest.approx(0.002857142857142857, rel=1e-6)


def test_set_initial_condition():
    """
    Checks function SolveDiffusion2D.get_initial_function
    """
    solver = SolveDiffusion2D()
    solver.nx = 100
    solver.ny = 100
    solver.dx = 0.1
    solver.dy = 0.1
    solver.T_cold = 300.0
    solver.T_hot = 750.0

    u = solver.set_initial_condition()


    u_test = solver.T_cold * np.ones((solver.nx, solver.ny))
    r, cx, cy = 2, 5, 5
    r2 = r ** 2
    for i in range(solver.nx):
        for j in range(solver.ny):
            p2 = (i * solver.dx - cx) ** 2 + (j * solver.dy - cy) ** 2
            if p2 < r2:
                u_test[i, j] = solver.T_hot

    ground_truth = u_test.copy()
    assert np.array_equal(ground_truth, u)