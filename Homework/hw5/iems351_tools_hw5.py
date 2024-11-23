import numpy as np
import pandas as pd

###################################################################################################################
# IEMS 351, HW 5, Fall 2024
# Please write your codes in the TODO blocks
###################################################################################################################


# gradient method
def gradient_method(x_init, Z, y, model, param):
    """

    :param x_init: numpy array (n,)
    :param Z: numpy array (N,n)
    :param y: numpy array (N,)
    :param model: dictionary
        model["grad"](x,Z,y) returns a gradient in numpy array (N,)
        model["obj"](x,Z,y) returns a objective function value in float
        model["prediction_accuracy"](x,Z,y) returns prediction accuracy in float
    :param param: dictionary
        param["max_iteration"]: integer
        param["alpha"]: float, stepsize
        param["flag_constant_step_size"]: boolean
        param["flag_diminishing_step_size"]: boolean
    :return:
    """
    x_cur = x_init
    x_next = x_cur
    # ======================================================================================
    # TODO: compute the gradient at x_cur
    g_cur = model["grad"](x_cur, Z, y)
    # ======================================================================================
    for iteration in range(param["max_iteration"]):
        # ======================================================================================
        # TODO: implement gradient method with constant stepsize and diminishing stepsize
        if param["flag_constant_step_size"]:
            x_next = x_cur - param["alpha"] * g_cur
        elif param["flag_diminishing_step_size"]:
            x_next = x_cur - (param["alpha"] / (iteration + 1)) * g_cur
        # ======================================================================================
        # ======================================================================================
        # TODO: update
        x_cur = x_next
        g_cur = model["grad"](x_cur, Z, y)
        # ======================================================================================
        # output objective function value. NOTE: I fixed the spelling of iteration
        if iteration % param["freq_print_obj"] == 0:
            print("*******************************************************************************")
            print("iteration: {}, objective: {}".format(iteration + 1, model["obj"](x_cur, Z, y)))
            print("*******************************************************************************")
        # output predictor accuracy
        if iteration % param["freq_print_accuracy"] == 0:
            print("===============================================================================")
            print("iteration: {}, accuracy: {}".format(iteration + 1, model["prediction_accuracy"](x_cur, Z, y)))
            print("===============================================================================")
    print(
        "Solution process of gradient method is finished. Output the objective function value of prediction accuracy."
    )
    print("iteration: {}, objective: {}".format(iteration + 1, model["obj"](x_cur, Z, y)))
    print("iteration: {}, accuracy: {}".format(iteration + 1, model["prediction_accuracy"](x_cur, Z, y)))
    return x_cur
