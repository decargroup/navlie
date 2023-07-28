import os
import sys

# Add the examples folder to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'examples'))

"""
examples/ex_batch_se3.py
examples/ex_batch_vector.py
examples/ex_ekf_vector.py
examples/ex_inertial_gps.py
examples/ex_inertial_nav.py
examples/ex_interacting_multiple_model_se3.py
examples/ex_interacting_multiple_model_vector.py
examples/ex_invariant_so3.py
examples/ex_iterated_ekf_se3.py
examples/ex_monte_carlo.py
examples/ex_random_walk.py
examples/ex_sequential_measurements.py
examples/ex_ukf_se2.py
examples/ex_ukf_vector.py
examples/ex_varying_noise.py
"""

def test_ex_batch_se3():
    from ex_batch_se3 import main
    main()

def test_ex_batch_vector():
    from ex_batch_vector import main
    main()

def test_ex_ekf_vector():
    from ex_ekf_vector import main
    main()

def test_ex_inertial_gps():
    from ex_inertial_gps import main
    main()

def test_ex_inertial_nav():
    import ex_inertial_nav

# def test_ex_imm_se3():
#     import ex_imm_se3

def test_ex_imm_vector():
    from ex_imm_vector import main
    main()

def test_ex_invariant_so3():
    from ex_invariant_so3 import main 
    main()

def test_ex_iterated_ekf_se3():
    from ex_iterated_ekf_se3 import main 
    main()

def test_ex_monte_carlo():
    from ex_monte_carlo import main 
    main()

def test_ex_random_walk():
    from ex_random_walk import main 
    main()

def test_ex_sequential_measurements():
    from ex_sequential_measurements import main
    main()

def test_ex_ukf_se2():
    from ex_ukf_se2 import main 
    main()

def test_ex_ukf_vector():
    from ex_ukf_vector import main
    main()

def test_ex_varying_noise():
    from ex_varying_noise import main
    main()

    