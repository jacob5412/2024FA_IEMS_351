import copy

import numpy as np
import scipy as sp

###################################################################################################################
# IEMS 351, HW 7, Fall 2024
# Please write your codes in the TODO blocks
###################################################################################################################


# factorize with mod for Newton's method
def factorize_with_mod(H, init_correction_factor, increase_factor):
    """
    :param H: symmetric numpy array, Hessian
    :param init_correction_factor: lambda in the slides of lecture 19
    :param increase_factor: c in the slides of lecture 19
    :return: L: numpy array, lower triangular matrix
            correction_factor: float
    """
    n_row, n_col = H.shape
    H_cp = copy.deepcopy(H)  # take a deep copy of H for Cholesky decomposition
    correction_factor = init_correction_factor
    I = np.eye(n_row)
    flag_pd = False
    it = 0
    while not flag_pd:
        try:
            # ======================================================================================
            # TODO Use np.linalg.cholesky to perform Cholesky decomposition
            L = np.linalg.cholesky(H_cp)
            # ======================================================================================
        except np.linalg.LinAlgError:
            flag_pd = False
        else:
            flag_pd = True
        if not flag_pd:
            # ======================================================================================
            # TODO increase the correction factor and modify the hessian matrix
            correction_factor *= increase_factor
            H_cp = H_cp + correction_factor * I
            # ======================================================================================
            it += 1
    if it == 0:
        return L, 0
    return L, correction_factor


# backtracking line search
def backtracking_linesearch(x, d, func_val_cur, grad_mul_d, model, param):
    """

    :param x: numpy array, current iterate
    :param d: numpy array, search direction
    :param func_val_cur: float, objective function value at the current iterate
    :param grad_mul_d: float, inner product of gradient and direction
    :param model: dictionary
           param["decrease_factor"]: descrease factor
           param["eta"]: relaxation parameter for the relaxed tangent line
    :param param: dictionary
    :return: x_next: numpy array, next iterate
            alpha_descent: float, step size to ensure sufficient descent
            flag_success: boolean, tell if backtracking line search is successful or not
    """
    alpha_descent = param["alpha_constant"]
    flag_success = True
    x_next = x + alpha_descent * d
    f_along_line = model["func"](x_next)
    f_relaxed_tangent = func_val_cur + alpha_descent * param["eta"] * grad_mul_d
    it_count = 0
    while f_along_line - f_relaxed_tangent > param["tol"]:
        # ======================================================================================
        # TODO Update the step size, function values along the line and relaxed tangent line at the new iterate
        # reduce the step size
        alpha_descent *= param["decrease_factor"]
        x_next = x + alpha_descent * d
        f_along_line = model["func"](x_next)
        f_relaxed_tangent = func_val_cur + alpha_descent * param["eta"] * grad_mul_d
        # ======================================================================================
        it_count += 1
        if it_count > param["max_it_backtracking"]:
            flag_success = False
            return x_next, alpha_descent, flag_success
    return x_next, alpha_descent, flag_success


# Newton's method
def newton_method_minimize(x_init, model, param):
    """
    :param x_init: numpy array, initial estimated solution
    :param model: dictionary,
        model["func"](x) returns a function value in float
        model["grad"](x) returns a gradient in numpy array
        model["hessian"](x) returns a Hessian matrix in numpy array
    :param param: dictionary,
        param["decrease_factor"]: float, < 1
        param["increase_factor"]: float, > 1
        param["correction_factor"]: float, > 0
        param["eta"]: float, relaxation parameter for the relaxed tangent line
        param["max_iteration"]: int, maximum number of iterations
        param["tol"]: float, tolerance level
        param["linesearch"]: string, type of line search (constant or backtracking)
        param["alpha_constant"]: float, step size
    :return:
        x_cur: numpy array, estimated solution
    """
    x_cur = x_init
    x_next = x_cur

    # compute gradient and Hessian
    func_val_cur = model["func"](x_cur)
    g_cur = model["grad"](x_cur)
    norm_g_cur = np.linalg.norm(g_cur)
    hessian_cur = model["hessian"](x_cur)
    print("Line Search Type: " + param["linesearch"])
    print(
        "iter    f                  ||d_k||            alpha               perturb             ||grad||      linesearch_success"
    )
    print(
        "0       {:.4e}        {:.4e}          {:.4e}         {:.4e}            {:.4e}        n/a".format(
            func_val_cur, 0, 0, 0, norm_g_cur
        )
    )
    for iteration in range(param["max_iteration"]):
        if norm_g_cur < param["tol"]:
            print("L-2 norm of the gradient is 0, a stationary point is found.")
            break
        # ======================================================================================
        # TODO Cholesky factorization
        L, correction_factor = factorize_with_mod(hessian_cur, param["correction_factor"], param["increase_factor"])
        # ======================================================================================
        # ======================================================================================
        # TODO use sp.linalg.solve_triangular to solve L v = - g_cur
        v = sp.linalg.solve_triangular(L, -g_cur, lower=True)
        # TODO use sp.linalg.solve_triangular to solve L^\top d = v
        d = sp.linalg.solve_triangular(L.T, v, lower=False)
        # ======================================================================================
        norm_d = np.linalg.norm(d)
        grad_mul_d = g_cur @ d
        flag_success = False
        alpha_descent = param["alpha_constant"]
        # checking if g^\top d < 0
        if grad_mul_d < 0:
            # line search
            if param["linesearch"] == "constant":
                x_next = x_cur + param["alpha_constant"] * d
            elif param["linesearch"] == "backtracking":
                # ======================================================================================
                # TODO Backtracking Line Search
                x_next, alpha_descent, flag_success = backtracking_linesearch(
                    x_cur, d, func_val_cur, grad_mul_d, model, param
                )
                # ======================================================================================
                if not flag_success:
                    x_next = x_cur - alpha_descent * g_cur
        else:  # switch to gradient method
            x_next = x_cur - alpha_descent * g_cur
            flag_success = False
        # update
        x_cur = x_next
        func_val_cur = model["func"](x_cur)
        g_cur = model["grad"](x_cur)
        norm_g_cur = np.linalg.norm(g_cur)
        hessian_cur = model["hessian"](x_cur)
        print(
            "{}       {:.4e}        {:.4e}          {:.4e}         {:.4e}            {:.4e}        {}".format(
                iteration + 1, func_val_cur, norm_d, alpha_descent, correction_factor, norm_g_cur, flag_success
            )
        )
    return x_cur
