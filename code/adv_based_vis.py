from art import config
from art.estimators.classification import BlackBoxClassifier, BlackBoxClassifierNeuralNetwork
from art.attacks.evasion import *
from art.utils import to_categorical
from art.utils import load_dataset, get_file, compute_accuracy
import pickle
from helper import *
from KitNET.KitNET import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from integrated_gradients_bb import integrated_gradients
from scipy.stats import binned_statistic_2d
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import sys
import os
import logging
import itertools

rng = np.random.default_rng(42)


def setup_logger(name, log_file, logger_level=logging.WARNING):
    """To setup as many loggers as you want"""
    # logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    logger = logging.getLogger(name)
    logger.setLevel(logger_level)

    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logger_level)

    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)

    # logger.addHandler(logging.FileHandler(log_file,mode="w"))

    return logger


def feature_to_2d(x_0, features, q, func=None, feature_range=None):
    if feature_range is None:
        diff_vector = (features-x_0)
    else:
        diff_vector = (features-x_0)/feature_range
    result, residual, _, _ = np.linalg.lstsq(q, diff_vector.T, rcond=None)

    if feature_range is None:
        plane_out = x_0+np.einsum("ki,lk->il", result, q)
    else:
        plane_out = x_0+np.einsum("ki,lk,l->il", result, q, feature_range)

    plane_out = plane_out.reshape([-1, q.shape[0]])

    feature_diff = np.sum(np.abs(plane_out-features), axis=1)
    if np.max(residual) > 1e-10:
        print(feature_diff.shape)
        print(f"max feature diff {np.max(feature_diff)}")
        print(f"max residual {np.max(residual)}")
        print(f"max idx {np.argmax(residual)}")

    if func:
        pred_val = func(plane_out)
        true_val = func(features)

        return result, pred_val, true_val
    else:
        return result


def polygon_area(x, y):
    if x.size==0:
        return 0
    correction = x[-1] * y[0] - y[-1] * x[0]
    main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    return 0.5*np.abs(main_area + correction)


def plot_decision_boundary_3p(func, boundary_file, plot_range=None, half_way=None, symbols=None,
                              threshold=1, file_name="test", feature_range=None, plot_contour=True, 
                              ):
    if plot_contour:
        fig = make_subplots(rows=1, cols=2, specs=[
                            [{"type": "surface"}, {"type": "scattergl"}]])
        

    if isinstance(boundary_file,str):
        boundary_file = np.genfromtxt(boundary_file, delimiter=",")

    x_0=boundary_file[np.newaxis, 0]
    
    boundary, A=np.split(boundary_file, [-2])
    # plot decision boundary
    boundary_result, pred_val, true_val = feature_to_2d(
        x_0, boundary, A.T, func, feature_range)

    # reorder boundary and estimate area
    # remove last two
    ordered_boundary = boundary_result[:, :-2]
    # reorder boundary points
    if half_way:
        ordered_boundary = np.hstack(
            (ordered_boundary[:, :half_way-1], ordered_boundary[:, half_way:-1][:, ::-1]))

    area = polygon_area(ordered_boundary[0], ordered_boundary[1])

    if plot_contour:
        fig.add_trace(go.Scatter3d(x=boundary_result[0], y=boundary_result[1], z=true_val,  name="boundary", mode="markers",
                                    hovertext=[
            f"index: {i}, plane_as: {val}, true_as: {true_val[i]}" for i, val in enumerate(pred_val)],
            marker=dict(size=5,
                        color=true_val,
                        symbol=symbols,
                        colorscale='greys',
                        line=dict(
                            color='white',
                            width=3
                        )
                        ),
        ), row=1, col=1)

        fig.add_trace(go.Scattergl(x=boundary_result[0], y=boundary_result[1], name="boundary", mode="markers",
                                    hovertext=[
            f"index: {i}, plane_as: {val}, true_as: {true_val[i]}" for i, val in enumerate(pred_val)],
            marker=dict(size=10,
                        color=true_val,
                        opacity=0.5,
                        symbol=symbols,
                        line=dict(
                            color='MediumPurple',
                            width=2
                        )
                        ),
        ), row=1, col=2)

    # plot contour
    if plot_range is None:
        dir1 = np.linspace(np.min(
            boundary_result[0])-np.ptp(boundary_result[0])*0.05, np.max(boundary_result[0])+np.ptp(boundary_result[0])*0.05, 200)
        dir2 = np.linspace(np.min(
            boundary_result[1])-np.ptp(boundary_result[1])*0.05, np.max(boundary_result[1])+np.ptp(boundary_result[1])*0.05, 200)
    elif len(plot_range) == 1:
        dir1 = np.linspace(*plot_range[0])
        dir2 = np.linspace(*plot_range[0])
    else:
        dir1 = np.linspace(*plot_range[0])
        dir2 = np.linspace(*plot_range[1])

    xv, yv = np.meshgrid(dir1, dir2)

    coord_mat = np.dstack([xv, yv])

    if feature_range is None:
        input_val = x_0+np.einsum("ijk,kl->ijl", coord_mat, A)

    else:
        input_val = x_0+np.einsum("ijk,kl,l->ijl", coord_mat, A, feature_range)

    input_val = input_val.reshape([-1, A.shape[1]])
    input_val = np.nan_to_num(input_val)
    f_val = func(input_val)
    f_val = f_val.reshape(xv.shape)
    if plot_contour:
        fig.add_trace(go.Surface(z=f_val, x=dir1, y=dir2,
                                 contours={"z": {"show": True,
                                                 "project_z": True,
                                                 "start": 0, "end": threshold * 1.1,
                                                 "size": (threshold)/10.},
                                           "x": go.surface.contours.X(highlight=False),
                                           "y": go.surface.contours.Y(highlight=False)},

                                 showlegend=True), row=1, col=1)

        fig.add_trace(
            go.Contour(
                z=f_val,
                x=dir1,  # horizontal axis
                y=dir2,  # vertical axis
                opacity=0.4,
                line_smoothing=0,
                contours=dict(
                    showlabels=True,  # show labels on contours
                    labelfont=dict(  # label font properties
                        size=12,
                        color='white',
                    ),
                    start=np.min(f_val), end=threshold * 1.1, size=np.abs(threshold-np.min(f_val))/10.
                )
            ), row=1, col=2)

        fig.update_layout(
            width=1600,
            height=800
        )

        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
            row=1, col=2
        )
        fig.update_layout(
            scene_aspectmode='manual', scene_aspectratio=dict(x=1, y=1, z=1),

        )

        # # reverse order
        fig.data = fig.data[::-1]

        # fig.write_html(f"exp_figs/db_vis/{file_name}.html")

        fig.write_html(f"exp_figs/db_vis/{file_name}_3d.html")

    return f_val, area


def binary_search(target, original, threshold, func, logger=None):
    batch_size = original.shape[0]
    start = np.zeros((batch_size,))
    end = np.ones((batch_size,))
    # positive_midx = np.copy(target)

    logger.info(f"begin binary search")

    while np.all((end - start) > threshold):
        m = (end + start) / 2
        mid_x = m * target + (1 - m) * original

        mask, score = func(mid_x, True)
        logger.info(f"middle score {score}, m{m}")
        end[mask] = m[mask]
        start[~mask] = m[~mask]

        # positive_midx[mask] = mid_x[mask]

    return mid_x, mask


def linear_sample(x_start, x_end, samples):
    all_steps = np.linspace(0, 1, samples).astype(
        "float32")[1:-1][:, np.newaxis, np.newaxis]
    lin_samples = all_steps * \
        x_start[np.newaxis, :, :] + (1 - all_steps) * x_end[np.newaxis, :, :]
    return np.reshape(lin_samples, [-1, x_start.shape[1]])


def take_step(start, step, direction, feature_range=None):
    if feature_range is None:
        return start+np.einsum("j,jk->jk", step, direction)
    else:
        return start+np.einsum("j,jk,k->jk", step, direction, feature_range)


def linear_search(direction, original, step_size, threshold, func, logger, diff_threshold=1e-3, 
                  feature_range=None, max_iter=50):

    # ensure direction is unit vector
    logger.info("begin linear search")
    direction /= np.linalg.norm(direction)

    step = np.zeros(step_size.shape)
    count = np.zeros(step_size.shape)

    prev_label, scores = func(original, True)
    
    if np.all(np.abs(threshold - scores)<=diff_threshold):
        logger.info(f"initial score {scores}, already at boundary")
        return original, scores, step


    while np.any(np.abs(threshold - scores) > diff_threshold):
        step += step_size
        
        search_x = take_step(original, step, direction, feature_range)
        label, scores = func(search_x, True)

        # check if same label
        same_label = (label == prev_label)

        logger.debug(
            f"step {step}, scores at {count}: {scores}, same label: {same_label}, step size {step_size}")

        # if different label, move in the other direction and decrease step size
        step_size[~same_label] *= -0.5
        count[~same_label] = 0

        # update previous label and score
        prev_label = label
        count += 1
        
        if np.all(count >= max_iter):
            break
    logger.info("-"*50)
    
    return search_x, scores, step


def gen_random_adv_sample2(x_star, feature_range, func, threshold):
    # random normalized direction
    sample = np.copy(x_star)
    direction = rng.uniform(-1, 1, size=x_star.shape).astype("float32")

    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    step_size = 0.05
    sample = sample + step_size * direction * feature_range
    score = func(sample)
    count = 0
    max_count = 20
    while np.any(score < threshold + 1e-3):
        update_idx = score < threshold + 1e-3
        sample[update_idx] = sample[update_idx] + \
            step_size * direction[update_idx] * feature_range
        score[update_idx] = func(sample[update_idx])
        step_size += 0.05
    return sample, score


def gen_random_adv_sample(x_star, feature_range, func):
    upper_pert = 0.1

    adv_ex = rng.uniform(-upper_pert,
                         upper_pert, size=x_star.shape) * feature_range + x_star
    is_adv, scores = func(adv_ex, True)
    max_iter = 100
    iter = 0
    while not np.all(is_adv):
        adv_ex[~is_adv] = rng.uniform(-upper_pert,
                                      upper_pert, size=x_star[~is_adv].shape) * feature_range + x_star[~is_adv]
        label, score = func(adv_ex[~is_adv], True)

        scores[~is_adv] = score
        is_adv[~is_adv] = label
        if iter > max_iter:
            upper_pert += 0.1
            iter = 0
        iter += 1

    return adv_ex, scores


def perpendicular_vector(vector, direction=None):
    if direction is not None:
        if direction.ndim == 1:
            direction = np.tile(direction, (vector.shape[0], 1))
    else:
        direction = rng.uniform(-1, 1, vector.shape).astype("float32")

    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    # dot_prod = np.einsum('ij,ij->i', rand_vec[:, :-1], vector[:, :-1])
    #
    # rand_vec[:, -1] = -dot_prod / vector[:, -1]
    # rand_vec /= np.linalg.norm(rand_vec, axis=1, keepdims=True)
    dot_prod = np.einsum('ij,ij->i', direction, vector) / \
        np.einsum('ij,ij->i', vector, vector)
    dot_prod = dot_prod[:, np.newaxis] * vector

    perp = direction - dot_prod
    return perp


def intersection_vector(A, normal, prev, logger=None):
    dot_prod = np.einsum("ij,ki->kj", A, normal)

    normal /= np.linalg.norm(normal, axis=1)
    dot1 = dot_prod[:, 0]
    dot2 = dot_prod[:, 1]

    # if close to zero, return previous value
    if (np.abs(dot1) < 1e-9).any() and (np.abs(dot2) < 1e-9).any():
        if logger is not None:
            logger.info(f"boundary == tangent")
        return prev

    # elif (np.abs(dot1)<1e-9):
    #     return -A[None,:,0]
    # elif (np.abs(dot2)<1e-9):
    #     return -A[None,:,1]

    coef = np.vstack([np.full((normal.shape[0]), 1), -dot1/dot2])

    direction = np.einsum("ij, jk->ki", A, coef)
    if logger is not None:
        logger.info(f"dot1 {dot1}, dot2 {dot2}")

    # normalise direction
    direction /= np.linalg.norm(direction)

    if np.abs(angle(prev, direction)-np.pi) < 0.5:
        direction *= -1

    return direction


def modified_hsj(baseline_kitsune, scaler_path, benign_path, adv_path, threshold, num_samples=10, search_only=False):
    with open(baseline_kitsune, "rb") as m:
        baseline_kitsune = pickle.load(m)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    def decision(x, return_score=False):
        scores = baseline_kitsune.process(x)

        if return_score:
            return scores > threshold, scores
        else:
            return scores > threshold

    def scores(x):
        scores = baseline_kitsune.process(x)

        return scores

    def scores_cost(x):
        scores = baseline_kitsune.process(x)
        scores = 1 - np.abs(scores - threshold)
        return scores

    feature_range = (scaler.data_max_ - scaler.data_min_).numpy()

    dataset = get_dataset(benign_path, 1, frac=1, shuffle=False,
                          read_with="tf", seed=None, dtype="float32")
    found = 0
    max_iter = 1000
    max_steps = 10
    batch_size_random_sample = 64
    theta = 1e-4
    count = 0
    with open(adv_path, "w") as adv_f:
        for x_star in tqdm(dataset):
            x_star = x_star.numpy()
            np.savetxt(adv_f, x_star, delimiter=',')

            # prev_d = np.linalg.norm(prev_x_hat - x_star, axis=1, keepdims=True)
            prev_x_hat, _ = gen_random_adv_sample(
                x_star, feature_range, decision)

            for t in range(max_iter):

                # np.savetxt(adv_f, prev_x_hat, delimiter=',')

                x_t, label = binary_search(
                    prev_x_hat, x_star, theta, decision, logger=logger)
                if t >= 1:
                    dist = np.linalg.norm(scaler.transform(
                        x_t) - scaler.transform(prev_xt), axis=1)
                    print("dist", dist)

                np.savetxt(adv_f, x_t, delimiter=',')

                if search_only:
                    prev_x_hat = gen_random_adv_sample(
                        x_star, feature_range, decision)
                    continue

                B_t = int(batch_size_random_sample * np.sqrt(t + 1))
                # B_t = batch_size_random_sample

                delta_t = np.linalg.norm(scaler.transform(
                    x_star) - scaler.transform(prev_x_hat), axis=1, keepdims=True) / 100

                v_t = monte_carlo_estimate(
                    scores, x_t, B_t, delta_t, feature_range)

                step_size = np.linalg.norm(scaler.transform(
                    x_t) - scaler.transform(x_star), axis=1, keepdims=True) * 0.1

                new_label, a_score = decision(
                    x_t + step_size * feature_range * v_t, return_score=True)

                steps = 0
                print(decision(x_t, return_score=True), a_score)
                step_size[a_score < threshold] *= -1
                while np.any(a_score - threshold < 0) and steps < max_steps:
                    step_size[a_score - threshold < 0] /= 2.

                    new_label, a_score = decision(
                        x_t + step_size * feature_range * v_t, return_score=True)
                    steps += 1

                print("a score", a_score)
                if np.any(a_score - threshold < -1e-3):
                    print(t)
                    print("no adv sample")
                    break

                # for a, b, c, d in zip(label, new_label, step_size, a_score):
                #     print(
                #         f"original: {a}, new: {b}, step_size: {c}, score: {d}")
                # print("-" * 50)
                # raise Exception()
                prev_xt = x_t
                prev_vt = v_t
                prev_x_hat = x_t + step_size * feature_range * v_t
                # prev_d = np.linalg.norm(scaler.transform(
                #     x_t) - scaler.transform(x_star))
            count += 1
            if count == 1:
                break


def step_size_estimate(func, x_n, step_size, tangent_d, feature_range, err, logger=None):
    forward_sample = take_step(x_n, step_size, tangent_d, feature_range)
    backward_sample = take_step(x_n, step_size, -tangent_d, feature_range)
    
    y_val=func(np.vstack([backward_sample,x_n,forward_sample]))
    x_val=np.array([-step_size[0],0,step_size[0]])
    
    R=circle_radius(x_val,y_val)
    # R = (step_size**2+d**2)/(2*d)
    if logger is not None:
        logger.info(f"step size {step_size} err {err} estimated radius {R}")
        
    # return R*0.018
    return np.sqrt(2*err*R-err**2)


def curvature(func, x, grad, v=None):
    def grad_func(x):
        return gradient_estimate(func, x, 1e-6)

    H = gradient_estimate(grad_func, x, 1e-6)
    grad_mag = np.linalg.norm(grad, axis=1)
    if v is None:
        P = np.identity(grad.shape[0])-np.einsum("ij,ik->ijk", grad, grad)
        matrix = 1/grad_mag * np.einsum("mij,mjk,mkl->mil", P, H, P)
        return np.linalg.eigvals(matrix).real

    else:
        curvature = np.einsum("ij, ijk, ik->i", v, H, v) / \
            (np.einsum("ij,ij->i", v, v)*grad_mag)
        return curvature[:, np.newaxis]


def find_perpendicular_vector(directional_derivative, A, direction):
    tangent = np.hstack([directional_derivative[None, :, 1], -
                        directional_derivative[None, :, 0]])*direction
    tangent_d = np.einsum("ij,kj->ik", tangent, A)
    tangent_d /= np.linalg.norm(tangent_d, axis=1, keepdims=True)

    return tangent_d

def minor(m,i,j):
    return np.delete(np.delete(m, i, 0),j,1)

def circle_radius(x_val,y_val):
    col1=x_val**2+y_val**2
    A=np.vstack([col1,x_val,y_val,np.ones(3)]).T

    M11=np.linalg.det(np.delete(A, 0,1))
    if M11==0:
        return 0.1
    M12=np.linalg.det(np.delete(A, 1,1))
    M13=np.linalg.det(np.delete(A, 2,1))
    M14=np.linalg.det(np.delete(A, 3,1))
    x0=0.5*M12/M11
    y0=-0.5*M13/M11 
    
    r=x0**2+y0**2+M14/M11 
    return np.sqrt(r)

def boundary_traversal(baseline_nids, start_position, plane, output_boundary_path,
                       dr_model=None, run_name="", max_iter=500, max_step=1,
                       logger_level=None, eps=1e-3, draw_plots=False, idx=None, write=True):

    # set up logger
    threshold = baseline_nids.threshold

    if logger_level is None:
        logger = logging.getLogger('dummy')
    else:
        logger = setup_logger(run_name,
                              f'boundary_logs/{run_name}/{idx}.log', logger_level)

    feature_range = (scaler.data_max_ - scaler.data_min_)

    traced_boundary=[]

    if not isinstance(start_position, np.ndarray):
        start_position = start_position.numpy()
    if start_position.ndim == 1:
        start_position = np.expand_dims(start_position, axis=0)

    start_label, start_score = baseline_nids.decision(start_position, True)

    traced_boundary.append(start_position)

    x_n_score = start_score
    x_n = np.copy(start_position)
    statistics = {"score": start_score[0],
                    "drawn":draw_plots,
                    "init": True,
                    "init_dist":0,
                    "irregular": 0, "jagged": 0, "failed": 0, "discontinuous": 0, "distance": 0, "complete": 0, "enclosed": False}

    # find the search direction
    if isinstance(plane, np.ndarray):
        v = start_position-plane
        
    elif plane == "random":
        v = rng.uniform(scaler.data_min_,
                            scaler.data_max_, (2,start_position.shape[1]))
        
    elif plane == "grad_attrib":
        median_baseline = np.array([[1.77647977e+01,  7.91110919e+02,  2.69185140e+02,  2.89514298e+01,
                                    8.08515834e+02,  1.25510713e+03,  8.41333078e+01,  8.28662326e+02,
                                    9.97855458e+03,  8.11220971e+02,  8.43755024e+02,  2.37963303e+05,
                                    5.23390208e+03,  8.58700374e+02,  2.72830619e+05,  1.77647977e+01,
                                    7.91110919e+02,  2.69185140e+02,  8.93315557e+02,  2.66163433e+05,
                                    6.72632935e-04,  5.35686830e-04,  2.89514298e+01,  8.08515834e+02,
                                    1.25510713e+03,  8.93028176e+02,  2.70974586e+05,  2.88202631e-02,
                                    2.24825751e-03,  8.41333078e+01,  8.28662326e+02,  9.97855458e+03,
                                    8.93862228e+02,  2.77694343e+05,  0.00000000e+00,  0.00000000e+00,
                                    8.11220971e+02,  8.43755024e+02,  2.37963303e+05,  8.96559861e+02,
                                    2.85847708e+05, -9.54228266e+00, -2.98488341e-03,  5.23390208e+03,
                                    8.58700374e+02,  2.72830619e+05,  9.08172132e+02,  2.92552247e+05,
                                    -4.68353220e+01, -8.55576133e-03,  1.76916091e+01,  1.69885399e-02,
                                    1.29984952e-04,  2.88510972e+01,  1.70497534e-02,  1.32537505e-04,
                                    8.38732978e+01,  1.72235522e-02,  1.36116072e-04,  8.06415332e+02,
                                    1.76861989e-02,  1.93991241e-04,  4.80839061e+03,  2.16405462e-02,
                                    1.81350137e-02,  1.75268811e+01,  7.75385079e+02,  1.92532352e+00,
                                    8.98608245e+02,  2.59262226e+05,  2.85351986e-04,  3.90553909e-04,
                                    2.85986391e+01,  7.73559937e+02,  3.41993084e+00,  8.98503484e+02,
                                    2.64061834e+05,  1.18725609e-02,  1.67338817e-03,  8.31752317e+01,
                                    7.64892930e+02,  3.47107472e+01,  9.00049801e+02,  2.69967382e+05,
                                    0.00000000e+00,  0.00000000e+00,  7.99036398e+02,  7.59354482e+02,
                                    3.13542706e+02,  9.01881353e+02,  2.71620228e+05,  0.00000000e+00,
                                    0.00000000e+00,  4.62262446e+03,  7.56464738e+02,  6.15421856e+02,
                                    8.96685506e+02,  2.68274320e+05,  0.00000000e+00,  0.00000000e+00]])
        # median_baseline=np.array([nids_model.min_feature["Cam_1"]])
        # find feature attributions of start to x_n
        v, _ = integrated_gradients(
            dr_model.transform, median_baseline, start_position, m_steps=128, recon=True)
        v=np.squeeze(v)
        
    else:
        raise ValueError(
            "plane must be two vectors, random or grad-attrib")

    
    v/=feature_range
    
    cosine = angle(v[None, 0], v[None, 1])
    statistics["v1v2angle"] = cosine
    
    A, _ = np.linalg.qr(v.T)
    
    #ensure the first vector have the same sign
    if np.sign(A[0,0])!=np.sign(v[0,0]):
        A*=-1
    
    plot_range=None

    init_step = 0.05
    
    x_init, x_init_score, init_dist_to_db = linear_search(A[None, :, 0], x_n, np.full(
        start_score.shape, init_step), threshold,
        baseline_nids.decision, diff_threshold=eps, logger=logger, 
        feature_range=feature_range, max_iter=400, 
        )
    start_coord=np.array([[1,0]])
    if np.abs(x_init_score-threshold)>eps:
        x_init, x_init_score, init_dist_to_db = linear_search(-A[None, :, 0], x_n, np.full(
        start_score.shape, init_step), threshold,
        baseline_nids.decision, diff_threshold=eps, logger=logger, 
        feature_range=feature_range, max_iter=400, 
       )
        start_coord=np.array([[-1,0]])
    
    if np.abs(x_init_score-threshold)>eps:
        x_init, x_init_score, init_dist_to_db = linear_search(A[None, :, 1], x_n, np.full(
        start_score.shape, init_step), threshold,
        baseline_nids.decision, diff_threshold=eps, logger=logger, 
        feature_range=feature_range, max_iter=400, 
       )
        start_coord=np.array([[0,1]])
        
    
    if np.abs(x_init_score-threshold)>eps:
        x_init, x_init_score, init_dist_to_db = linear_search(-A[None, :, 1], x_n, np.full(
        start_score.shape, init_step), threshold,
        baseline_nids.decision, diff_threshold=eps, logger=logger, 
        feature_range=feature_range, max_iter=400, 
       )
        start_coord=np.array([[0,-1]])


    if np.abs(x_init_score-threshold) > eps:
        statistics["init"] = False
        plot_range=[[-5,5,200]]
        statistics["drawn"]=True
    else:
        statistics["init_dist"]=init_dist_to_db[0]
    
    traced_boundary.append(x_init)

    direction = 1
    symbols = ["diamond", "diamond"]

    prev_angle = 0
    early_stop_flag = False
    fail_counter = 0
    half_way = None
    end_coord=None
    half1=False 
    half2=False

    x_n = np.copy(x_init)
    x_n_score = np.copy(x_init_score)
        

    if logger_level:
        logger.info(f"traversing {idx}")
        logger.info("="*50)
        logger.info(f"x start score: {start_score}, label: {start_label}")
        logger.info(f"initial boundary score: {x_n_score}")
        logger.info("*"*50)
    if statistics["init"]:
        for t in tqdm(range(max_iter)):
            symbol = "circle"
            if early_stop_flag:
                if direction == -1:
                    break
                half_way = t+1
                direction = -1
                x_n = x_init
                x_n_score = x_init_score
                early_stop_flag = False
                prev_angle = 0

            grad_f = gradient_estimate(baseline_nids.predict,
                                        x_n, delta_t=1e-5, direction=A.T, feature_range=feature_range)
            tangent_d = find_perpendicular_vector(grad_f, A, direction)

            # curvature calculation
            step_size = step_size_estimate(baseline_nids.predict, x_n, np.array(
                [1e-2]), tangent_d, feature_range, np.array([1e-4]), logger=logger)
            step_size = np.clip(step_size, 0.01, max_step)

            # take a step and get score
            x_np1 = take_step(x_n, step_size, tangent_d, feature_range)
            x_np1_score = baseline_nids.predict(x_np1)

            if logger_level:
                logger.info(f"step size est {step_size}")
                logger.info(f"after step {t+2} score {x_np1_score}")

                # find coordinate in boundary plane
                coord = (x_np1-start_position)/feature_range
                coords, res, _, _ = np.linalg.lstsq(
                    A, np.hstack([coord.T, tangent_d.T]), rcond=None)
                logger.info(
                    f"tangent direction: {coords[:,1]}, coordinate after step: {coords[:,0]}, res {res}")

            # correction step
            if np.any(np.abs(threshold - x_np1_score) > eps):
                logger.info("correct along gradient")
                # save uncorrected
                # traced_boundary.append(x_np1)
                # symbols.append("square-open")

                update_idx = np.abs(threshold - x_np1_score) > eps

                correction_dir = np.sign(
                    threshold-x_np1_score)*np.einsum("ij,kj->ik", grad_f, A)

                points, scores,_ = linear_search(
                    correction_dir, np.copy(
                        x_np1[update_idx]), step_size/20., threshold,
                    baseline_nids.decision, diff_threshold=eps,
                    logger=logger, feature_range=feature_range, max_iter=20,
                                        )

                # correct along -tangent
                if np.any(np.abs(threshold - scores) > eps):
                    # save uncorrected
                    # traced_boundary.append(points)
                    # symbols.append("square-open")

                    if logger_level:
                        logger.info(
                            "initial correction failed, search along -tangent")

                    points, scores,_ = linear_search(
                        -tangent_d, np.copy(points[update_idx]), step_size/20., threshold,
                        baseline_nids.decision, diff_threshold=eps,
                        logger=logger, feature_range=feature_range, max_iter=40,
                        )
                    symbol = "square"

                # correct along x_n
                if np.any(np.abs(threshold - scores) > eps):
                    if logger_level:
                        logger.info("correct along -gradient")
                    # save uncorrected
                    # traced_boundary.append(points)
                    # symbols.append("square-open")

                    points, scores,_ = linear_search(
                        -correction_dir, np.copy(points[update_idx]), step_size/20., threshold,
                        baseline_nids.decision, diff_threshold=eps,
                        logger=logger, feature_range=feature_range,
                        max_iter=20)
                    symbol = "cross"

                if np.any(np.abs(threshold - scores) > eps):
                    symbol = "x"
                    if logger_level:
                        logger.warning(
                            "cannot find boundary, incorrect gradient, revert back to x_n")
                    points = x_n
                    scores = x_n_score

                x_np1[update_idx] = points
                x_np1_score[update_idx] = scores

                logger.info(f"end_score {x_np1_score}")

            if symbol == "circle":
                max_step = np.minimum(0.1, max_step*2)
                if fail_counter > 0:
                    statistics["discontinuous"] += 1
                fail_counter = 0

            else:
                max_step = np.maximum(1e-3, max_step*0.5)
                fail_counter += 1
                if symbol == "x":
                    statistics["failed"] += 1
                    symbols[-1] = "x"
                    continue
                elif symbol == "square":
                    statistics["irregular"] += 1
                elif symbol == "cross":
                    statistics["jagged"] += 1

            x_np1_coord, _, _, _ = np.linalg.lstsq(
                A, ((x_np1-start_position)/feature_range).T, rcond=None)

            dist_from_prev = ln_distance(
                x_np1/feature_range, x_n/feature_range, 2)[0, 0]
            if logger_level:
                logger.debug(f"symbol {symbol} counter {fail_counter}")
                logger.info(f"distance {dist_from_prev}")
                        
            statistics["distance"] += dist_from_prev

            symbols.append(symbol)

            traced_boundary.append(x_np1)

            x_n = np.copy(x_np1)
            x_n_score = np.copy(x_np1_score)

            if t == max_iter//2 and statistics["complete"] == 0:
                early_stop_flag = True
                if logger_level:
                    logger.info("bidirectional early stopping at half way")

            if logger_level:
                
                logger.debug(f"x_np1_coord {x_np1_coord.T}")
            theta = angle_clockwise(start_coord, x_np1_coord.T)[0]

            logger.info(
                f"theta {np.degrees(theta)}, prev angle {np.degrees(prev_angle)}")

                
            
            if end_coord is None:
                # check if point passes pi or 0
                cond=prev_angle != 0 and np.sign(prev_angle) != np.sign(theta)
                # if passing pi, we are half way
                if np.abs(theta)> np.pi/2:
                    half1=True
            else:
                # check if we have reached halfway point and we have moved sufficiently far away
                cond=t-half_way>10 and ln_distance(end_coord, x_n/feature_range, 2)<0.1 
                
            if cond:
                early_stop_flag = True
                logger.info("bidirectional early stopping")
                statistics["complete"] += 1
                # if stopping at pi
                
                end_coord=np.copy(x_n)/feature_range
            
            prev_angle = theta

            logger.info("*"*50)
        
        statistics["total"] = t
        
        if t==1999:
            statistics["drawn"]=True
        
    # save anchor and end points
    if isinstance(plane, np.ndarray):
        traced_boundary.append(plane)
        symbols.append("circle-open")
        symbols.append("circle-open")
        
    traced_boundary.append(A.T)
    
    traced_boundary=np.vstack(traced_boundary)
    # [[1.75,1.85,100],[0.45,0.55,100]]
    f_val, area = plot_decision_boundary_3p(baseline_nids.predict, traced_boundary,plot_range=plot_range, 
                                            symbols=symbols, threshold=threshold, feature_range=feature_range, half_way=half_way,
                                            file_name=f"{run_name}/{idx}", plot_contour=statistics["drawn"])
    statistics["plane_std"] = np.std(f_val)
    statistics["plane_pos"] = np.mean(f_val > threshold)
    statistics["area"] = area
    statistics["enclosed"] = half1 and (statistics["complete"]==2)

    if write:
        with open(f"{output_boundary_path}{idx}.csv", "w") as adv_f:
            np.savetxt(adv_f, traced_boundary,delimiter=',')
    
    if fail_counter >= 1:
        statistics["discontinuous"] += 1
    
    
    return statistics


def ln_distance(x1, x2, ord):
    return np.sum(np.abs(x1-x2)**ord, axis=1, keepdims=True)**(1./ord)


def angle(v1, v2, pairwise=True):
    v1 /= np.linalg.norm(v1, axis=1, keepdims=True)
    v2 /= np.linalg.norm(v2, axis=1, keepdims=True)
    if pairwise:
        einsum_str = 'ij,kj->ik'
    else:
        einsum_str = 'ij,ij->i'
    return np.squeeze(np.arccos(np.clip(np.einsum(einsum_str, v1, v2), -1.0, 1.0)))


def angle_clockwise(v2, v1):
    dot = v1[:, 0]*v2[:, 0] + v1[:, 1]*v2[:, 1]
    det = v1[:, 0]*v2[:, 1] - v1[:, 1]*v2[:, 0]
    angle = np.arctan2(det, dot)
    return angle


def sample_near_point(points, gradx, grady, feature_range, range=[0.5, 0.5], steps=[5, 5]):
    x = np.linspace(-range[0], range[0], steps[0])
    y = np.linspace(-range[1], range[1], steps[1])

    vx, vy = np.meshgrid(x, y)
    points = points[np.newaxis, np.newaxis, :, :] + gradx[np.newaxis, np.newaxis, :, :] *\
        vx[:, :, np.newaxis, np.newaxis] * feature_range[np.newaxis, np.newaxis, np.newaxis, :] +\
        grady[np.newaxis, np.newaxis, :, :] * \
        vy[:, :, np.newaxis, np.newaxis] * \
        feature_range[np.newaxis, np.newaxis, np.newaxis, :]
    return points.reshape([-1, points.shape[-1]])


def random_sample(baseline_kitsune, scaler_path, benign_path, adv_path, threshold, num_samples=10):
    with open(baseline_kitsune, "rb") as m:
        baseline_kitsune = pickle.load(m)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    feature_range = (scaler.data_max_ - scaler.data_min_).numpy()

    dataset = get_dataset(benign_path, 1024, frac=1,
                          read_with="tf", seed=None, dtype="float32")
    found = 0
    with open(adv_path, "a") as adv_f:
        for i in tqdm(dataset):
            i = i.numpy()
            i_shape = i.shape

            neighbours = rng.uniform(1, 1.5, size=(
                i_shape[0], num_samples, i_shape[1])) * np.expand_dims(i, axis=1)
            neighbours = neighbours.reshape((-1, 100))

            scores = baseline_kitsune.predict(neighbours)
            adv_idx = np.where(scores > threshold)
            found += len(adv_idx)
            adv_feat = neighbours[adv_idx]
            np.savetxt(adv_f, adv_feat, delimiter=',')
    print(f"found {found} adversarial exampls")


def create_adversarial_examples(baseline_kitsune, scaler_path, benign_path, adv_path, atk_type, threshold):

    with open(baseline_kitsune, "rb") as m:
        baseline_kitsune = pickle.load(m)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    def predict_logit(x, scores=False):

        x = np.array(x)
        x = np.reshape(x, (-1, 100))
        pred_score = baseline_kitsune.predict(x)

        benign_conf = threshold / (pred_score + threshold)
        mal_conf = 1 - benign_conf
        return tf.stack([benign_conf, mal_conf], axis=1)

        # return labels
        # return pred_score

    def predict_labels(x, scores=False):
        pred_score = baseline_kitsune.predict(x)

        labels = tf.keras.utils.to_categorical(
            tf.greater(pred_score, threshold), 2)
        return labels

    feature_range = (scaler.data_max_ - scaler.data_min_).numpy()

    dataset = get_dataset(benign_path, 1024, frac=1,
                          read_with="tf", seed=None, dtype="float32")

    with open(adv_path, "a") as adv_f:
        if atk_type == "hsj":
            classifier = BlackBoxClassifier(
                predict_labels, (100,), 2)
            attack = HopSkipJump(classifier=classifier, targeted=False, batch_size=1024,
                                 max_iter=10, max_eval=100, init_eval=10, verbose=True)

            for i in tqdm(dataset):
                target = np.random.uniform(0, 1, size=i.shape) * feature_range
                x_adv = attack.generate(x=i.numpy(), x_adv_init=target)
                # labels = predict(x_adv)
                # adv_idx = np.where(labels[:, 1] == 1)
                #
                # adv_feat = x_adv[adv_idx]
                np.savetxt(adv_f, x_adv, delimiter=',')
        elif atk_type == "sqa":
            clip_values = (np.reshape(scaler.data_min_, (10, 10, 1)),
                           np.reshape(scaler.data_max_, (10, 10, 1)))

            # Create blackbox object
            classifier = BlackBoxClassifierNeuralNetwork(
                predict_logit, (10, 10, 1), 2, channels_first=False, clip_values=clip_values)

            attack = SquareAttack(estimator=classifier, eps=1., batch_size=2048,
                                  verbose=False,  norm=2)

            for i in tqdm(dataset):
                i = tf.reshape(i, (-1, 10, 10, 1)).numpy()
                x_adv = attack.generate(x=i, verbose=True)
                adv_feat = np.reshape(x_adv, (-1, 100))
                np.savetxt(adv_f, adv_feat, delimiter=',')


def draw_db(dr_model, files, heat_map=-1, file_name="adv_boundary", bidirectional=True):
    if heat_map == -1:
        fig = make_subplots(rows=1, cols=1)

    all_x = []
    all_y = []
    all_recon = []

    for i, tup in enumerate(files.items()):

        name, config = tup

        baseline_nids = config["nids_model"]
        threshold = baseline_nids.threshold
        nids_name = baseline_nids.name

        latent, dr_recon = get_latent_position(
            dr_model, None, config["file_path"], frac=config["frac"], shuffle=config["shuffle"], batch_size=config["batch_size"], seed=42, dtype="float32", read_with="tf", skip_header=config["skip_header"])

        _, recon = get_latent_position(
            baseline_nids, None, config["file_path"], frac=config["frac"], shuffle=config["shuffle"], batch_size=config["batch_size"], seed=42, dtype="float32", read_with="tf", skip_header=config["skip_header"])

        if len(dr_recon) > 0:
            dr_recon = dr_recon.reshape([-1, config["plot_batch_size"]])
        latent = latent.reshape([-1, config["plot_batch_size"],  2])
        recon = recon.reshape([-1, config["plot_batch_size"]])
        max_iter = (latent.shape[0])//2

        if config["draw_type"] == "boundary":
            recon = recon[1:-1]
            latent = latent[1:-1]
            if len(dr_recon) > 0:
                dr_recon = dr_recon[1:-1]

            if bidirectional:
                if len(dr_recon) > 0:
                    dr_recon = np.vstack(
                        (dr_recon[max_iter:][::-1, :], dr_recon[:max_iter]))
                recon = np.vstack(
                    (recon[max_iter:][::-1, :], recon[:max_iter]))
                latent = np.vstack(
                    (latent[max_iter:][::-1, :], latent[:max_iter]))

                text = [f"{nids_name}:{threshold:.3f}" if i ==
                        max_iter else " " for i in range(latent.shape[0])]
            else:
                text = [f"{nids_name}:{threshold:.3f}" if i ==
                        0 else " " for i in range(latent.shape[0])]
        else:
            text = ""

        for j in range(latent.shape[1]):
            if i <= heat_map:
                all_x.append(latent[:, j, 0])
                all_y.append(latent[:, j, 1])
                all_recon.append(
                    np.where(recon[:, j] > threshold, 1, 0))
            if i == heat_map:
                x_pos = np.concatenate(all_x)
                y_pos = np.concatenate(all_y)
                z_score = np.concatenate(all_recon)

                ret = binned_statistic_2d(
                    x_pos, y_pos, z_score, statistic=gini_index, bins=[200, 300],)

                # mask invalid values
                data = np.ma.masked_invalid(ret.statistic.T)
                x_values = (ret.x_edge[1:] + ret.x_edge[:-1]) / 2
                y_values = (ret.y_edge[1:] + ret.y_edge[:-1]) / 2
                xx, yy = np.meshgrid(x_values, y_values)
                # get only the valid values
                x1 = xx[~data.mask]
                y1 = yy[~data.mask]
                newarr = data[~data.mask]

                # data = griddata((x1, y1), newarr.ravel(),
                #                 (xx, yy), method='cubic')

                fig = px.imshow(data, origin="lower",
                                color_continuous_scale="amp", x=x_values, y=y_values,
                                aspect='auto')
                # fig.update_traces(zsmooth="best")
            if i > heat_map:
                if config["draw_type"] == "adv_sample":
                    color = [config["start_color"]] + \
                        list(range((latent.shape[0] - 2) // 2)) + list(
                            range(0, -(latent.shape[0] - 2) // 2, -1)) + [config["end_color"]]
                    symbol = [config["symbol"]] + \
                        ["x" if i <
                         threshold - 1e-3 else "cross" if i > threshold + 1e-3 else "circle" for i in np.squeeze(recon[:, j])[:-2]] + [config["symbol"]]
                    opacity = [1.] + [config["opacity"]
                                      for _ in range(latent.shape[0] - 2)] + [1.]
                    mode = "markers"

                elif config["draw_type"] == "background":
                    color = ["blue" if i <
                             threshold - 1e-3 else "red" if i > threshold + 1e-3 else "green" for i in np.squeeze(recon[:, j])[:-2]]
                    symbol = [config["symbol"]] + \
                        ["circle" for _ in range(
                            latent.shape[0] - 2)] + [config["symbol"]]
                    opacity = config["opacity"]
                    mode = "markers"

                elif config["draw_type"] == "boundary":
                    mode = "lines+text"

                hovertext = [f"index: {k} <br> NIDS: {nids_name} AS: {recon[k,j]:.3f} <br> DR: {dr_model.name} AS: {dr_recon[k,j] if len(dr_recon)>0 else np.nan:.3f}" for k in range(
                    latent.shape[0])]

                fig.add_trace(go.Scattergl(x=latent[:, j, 0], y=latent[:, j, 1],
                                           name=name + f"{j}",
                                           opacity=0.5,
                                           hovertext=hovertext,
                                           text=text,
                                           textposition="bottom center",
                                           mode=mode,
                                           #    line=dict(color=threshold),
                                           marker=dict(
                    size=10,
                    colorscale=config["color_scale"],

                    # set color equal to a variable
                    # color=recon,
                    color=color,
                    symbol=symbol,
                    opacity=opacity
                )

                ))
                print(f"visualised {latent[:,j,0].shape} packets")

    # fig.update_yaxes(
    #     scaleanchor="x",
    #     scaleratio=1,
    # )

    fig.write_html(f"exp_figs/db_vis/{file_name}.html")
    print(file_name)


def gini_index(x):
    if x == []:
        return 1
    p_pos = np.sum(x) / len(x)
    return p_pos



def read_adversarial_file(dataset, network_atk, adv_atk, idx_offset, target_adv_idx=None, out_path=None):
    log_file = f"../../mtd_defence/datasets/{dataset}/adversarial/Cam_1/{network_atk}/{adv_atk}/logs/Cam_1_{network_atk}_iter_0.txt"
    adv_csv = f"../../mtd_defence/datasets/{dataset}/adversarial/Cam_1/{network_atk}/{adv_atk}/csv/Cam_1_{network_atk}_iter_0.csv"
    adv_ori_csv = f"../../mtd_defence/datasets/{dataset}/malicious/Cam_1/Cam_1_{network_atk}.csv"
    count = 0
    adv_idx = []
    mal_idx = []
    with open(log_file, "r") as f:
        record = False
        for line in f.readlines():
            if line.startswith("original"):
                record = True
                continue
            if line.startswith("mutation"):
                record = False
            if record:
                if target_adv_idx is None or count in target_adv_idx:
                    line = line.rstrip().split(",")
                    num_craft_pkt = int(float(line[6].split(" ")[1]))+1
                    start_idx = int(line[3]) - \
                        idx_offset - num_craft_pkt + 1
                    adv_idx.append(start_idx+num_craft_pkt)
                    mal_idx.append(int(line[2])+1)
                count += 1
            else:
                continue

    adv_features = pd.read_csv(
        adv_csv, skiprows=lambda x: x not in adv_idx, header=None).to_numpy()
    ori_adv_features = pd.read_csv(
        adv_ori_csv, skiprows=lambda x: x not in mal_idx, header=None, usecols=list(range(100))).to_numpy()

    if out_path is not None:
        with open(out_path, "w") as adv_f:
            np.savetxt(adv_f, np.hstack(
                [adv_features, ori_adv_features]), delimiter=',')
    return adv_idx, adv_features, ori_adv_features


def get_benign_sample(benign_path, idx):
    benign_sample = pd.read_csv(benign_path, usecols=list(
        range(100)), skiprows=idx, nrows=1, header=None)
    return benign_sample.to_numpy()


def sample_n_from_csv(path, n=None, row_idx=None, ignore_rows=0, total_rows=None, seed=42, header=None,**kwargs):
    if n is None and row_idx is None:
        raise Exception("n and row idx cannot both be none")

    if row_idx is None:
        rng = np.random.default_rng(seed)
        if total_rows is None:
            with open(path, "r") as fh:
                total_rows = sum(1 for row in fh)
        if (n > total_rows):
            raise Exception("n cannot be larger than total rows")

        row_idx = rng.choice(total_rows-ignore_rows, n,
                             replace=False)+ignore_rows
    row_idx = np.sort(row_idx)
    if path.endswith(".csv"):
        df = pd.read_csv(path, usecols=list(range(100)),
                        skiprows=lambda x: x not in row_idx, header=header)
        return row_idx, df.to_numpy()
    if path.endswith(".npy"):
        data=np.load(path)
        return row_idx, data[row_idx]
        
        
   


def get_closest_benign_sample(benign_path, example, transform_func=None, eps=1e-3):
    """finds the closest benign sample to adv sample

    Args:
        benign_path (string): path to benign samples
        adv_example (ndarray): adversarial samples with format malicious->craft->adversarial
        transform_func (function, optional): dimensionality reduction function. if supplied it will find the closest benign sample
        in the latent space, else it finds the closet benign sample in feature space. Defaults to None.

    Returns:
        np.ndarray: the closest benign sample
    """
    traffic_ds = get_dataset(benign_path, 1024,
                             scaler=None, frac=1, read_with="tf", dtype="float64",
                             seed=0, skip_header=True, shuffle=False)

    if transform_func is not None:
        example = transform_func(example)

    closest_dist = None
    nearest_sample = None
    closest_file_idx = None

    total = 0
    for data in tqdm(traffic_ds):
        data = data.numpy()
        if transform_func is not None:
            latent = transform_func(data)
        else:
            latent = data

        idx = np.arange(total, total+data.shape[0])+1

        distance = np.linalg.norm(
            latent[:,  None, :] - example[None, :, :], axis=-1)

        # ignore close distance
        distance = np.where(distance < eps, np.inf, distance)

        closest_idx = np.argmin(distance, axis=0, keepdims=True)

        current_closest_dist = np.take_along_axis(
            distance, closest_idx, axis=0)
        current_nearest_sample = data[closest_idx[0]]
        current_closest_file_idx = idx[closest_idx[0]]

        if closest_dist is None:
            closest_dist = current_closest_dist
            nearest_sample = current_nearest_sample
            closest_file_idx = current_closest_file_idx
        else:

            update_idx = (current_closest_dist < closest_dist)

            closest_dist[update_idx] = current_closest_dist[update_idx]
            closest_file_idx[update_idx[0]
                             ] = current_closest_file_idx[update_idx[0]]
            nearest_sample[update_idx[0]
                           ] = current_nearest_sample[update_idx[0]]

        total += data.shape[0]
    print("processed", total)
    return nearest_sample, closest_dist, closest_file_idx


def test_nids(target_nids, scaler_path, file_path):

    for name, config in target_nids.items():
        baseline_nids = config["path"]
        threshold = config["threshold"]

        with open(baseline_nids, "rb") as m:
            baseline_nids = pickle.load(m)

        print(f"model type: {type(baseline_nids)}")

        if isinstance(baseline_nids, KitNET):
            scaler = None
        else:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

        _, recon, _ = get_latent_position(
            baseline_nids, scaler, file_path, frac=1, shuffle=False, batch_size=1024, seed=42, dtype="float32", read_with="tf", skip_header=True)

        plt.plot(recon/threshold, label=name, alpha=0.2)
    plt.axhline(1, label="threshold")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig('exp_figs/benign_as.png')

def reservoir_sample(files, file_db, nids, n=1000):
    reservoir=[]
    idx=[]
    file_names=[]
    counter=0
    for file in files:
        ds=get_dataset(file_db[file]["path"],1024,False,None,1,seed=42,drop_reminder=False)

        counter2=0
        for data in tqdm(ds):
            data=data.numpy()
            scores=nids.predict(data)
            lower_idx=np.where((nids_model.threshold*0.9<scores) & (scores<nids_model.threshold*1.1))
            lower_data=data[lower_idx]
            
            if lower_data.size==0:
                counter+=data.shape[0]
                counter2+=data.shape[0]
                continue
            
            for i, sample in zip(lower_idx[0], lower_data): 
                
                if len(reservoir)<n:
                    reservoir.append(sample)
                    idx.append(counter2+i)
                    file_names.append(file)
                    
                else:
                    j = np.random.randint(0,counter+i)
                    
                    if j < n:
                        reservoir[j] = sample
                        idx[j]=counter2+i
                        file_names[i]=file
            counter+=data.shape[0]
            counter2+=data.shape[0]
    return np.array(idx), np.array(reservoir), file_names

if __name__ == '__main__':
    dataset = "uq"
    device = "Cam_1"
    mtd_model_path = f"../../mtd_defence/models/{dataset}/mtd/Cam_1/fm0_mm1_am20"
    scaler_type = "min_max"
    scaler_path = f"../../mtd_defence/models/uq/autoencoder/Cam_1_scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    sampling_method = None
    distinguish_start_end = sampling_method == None

    atk_type = "bt"
    # adv_path = f"../../mtd_defence/datasets/uq_network/adversarial/decision_boundary_adversarial/db_vis_0.01_20_3_False_pso0.5_None/csv/decision_boundary_adversarial_iter_0.csv"
    # ae_name = "min_max_mean_10.0_recon_loss_sw_loss_contractive_loss_adam_denoising_2d_0.001_double_recon"

    dr_models = {"pca": {"type": "hybrid", "name": "pca", "save_type": "pkl", "path": "../models/pca.pkl",
                         "func_name": "transform", "scaler": scaler, "threshold": 0.3},
                 "umap": {"type": "hybrid", "name": "umap", "save_type": "pkl", "path": "../models/umap.pkl",
                          "func_name": "transform", "scaler": scaler, "threshold": 0.3},
                 "ae": {"type": "hybrid", "name": "recon_loss_denoising", "save_type": "tf", "path": "../models/min_max_mean_10.0_recon_loss_adam_denoising_2d_0.001",
                        "func_name": "call", "dr_output_index": 0, "ad_output_index": 1, "dtype": "float32", "threshold": 0.3},
                 "lle": {"type": "dimensionality_reduction", "name": "lle", "save_type": "pkl", "path": "../models/lle.pkl",
                         "func_name": "transform", "scaler": scaler, },
                 }

    with open("configs/files.json","r") as f:
        file_db=json.load(f)

    bidirectional = True

    # "ACK": {"file_path": "../../mtd_defence/datasets/uq/malicious/Cam_1/Cam_1_ACK_Flooding.csv", "color_scale": "Picnic", "frac": 0.03, "batch_size": 1024, "plot_batch_size": 1, "skip_header": True},
    # "SYN": {"file_path": "../../mtd_defence/datasets/uq/malicious/Cam_1/Cam_1_SYN_Flooding.csv", "color_scale": "Picnic", "frac": 0.05, "batch_size": 1024, "plot_batch_size": 1, "skip_header": True},
    # "UDP": {"file_path": "../../mtd_defence/datasets/uq/malicious/Cam_1/Cam_1_UDP_Flooding.csv", "color_scale": "Picnic", "frac": 0.02, "batch_size": 1024, "plot_batch_size": 1, "skip_header": True},
    # "ACK_adv": {"file_path": "../../mtd_defence/datasets/uq/adversarial/Cam_1/ACK_Flooding/autoencoder_0.5_10_3_False_pso0.5/csv/Cam_1_ACK_Flooding_iter_0.csv", "color_scale": "Picnic", "frac": 0.1, "batch_size": 1024, "plot_batch_size": 1},
    # "SYN_adv": {"file_path": "../../mtd_defence/datasets/uq/adversarial/Cam_1/SYN_Flooding/autoencoder_0.5_10_3_False_pso0.5/csv/Cam_1_SYN_Flooding_iter_0.csv", "color_scale": "Picnic", "frac": 0.1, "batch_size": 1024, "plot_batch_size": 1},
    # "UDP_adv": {"file_path": "../../mtd_defence/datasets/uq/adversarial/Cam_1/UDP_Flooding/autoencoder_0.5_10_3_False_pso0.5/csv/Cam_1_UDP_Flooding_iter_0.csv", "color_scale": "Picnic", "frac": 0.1, "batch_size": 1024, "plot_batch_size": 1},
    # }

    n_samples = 1000
    seed = 42
    draw_prob=0.5
    
    
    _, benign_samples = sample_n_from_csv(
        **file_db["Cam_1"], n=n_samples*2, seed=seed+1)
    


    # explaining adversarial examples
    rng = np.random.default_rng(seed)
    
    benign_tuple = ("Cam_1", np.hstack([benign_samples[:n_samples,np.newaxis,:], benign_samples[n_samples:,np.newaxis,:]]))
    
    starts = ["Cam_1","All"]
    
    planes=[benign_tuple]
    
 
    ae=["denoising_autoencoder_sigmoid_2_D","autoencoder_relu_2_D" ,"autoencoder_sigmoid_2_D",
        "denoising_autoencoder_sigmoid_2_filtered_0.2","autoencoder_relu_2_filtered_0.2" ,"autoencoder_sigmoid_2_filtered_0.2",
        "kitsune","autoencoder_relu_2" ,"autoencoder_sigmoid_2","denoising_autoencoder_sigmoid_2","autoencoder_sigmoid_25"]
    epochs=["1","20","40","60","80","100"]
    target_nidses =  [f"{a}_{e}" for a, e in itertools.product(ae, epochs)]


    for start_name in starts:
        if start_name=="All":
            near_boundary=True
        
        else:
            near_boundary=False
            target_idx, start_samples = sample_n_from_csv(
            **file_db[start_name], n=n_samples, seed=seed)
            file_names=[start_name for i in range(n_samples)]
            
        
        for nids_model in target_nidses:
            nids_model = get_nids_model(nids_model, "opt_t") 
            if near_boundary:
                start_name="All"
                target_idx, start_samples, file_names=reservoir_sample(["Cam_1","ACK","SYN","UDP","PS","SD"], file_db, nids_model, n_samples)
                
            
            draw=np.random.choice(a=[False, True], size=target_idx.shape, p=[draw_prob, 1-draw_prob])  
            
             
            filename=f"exp_csv/db_characteristics/{nids_model.name}_{nids_model.threshold:.3f}_bt_results_{start_name}.csv"
            file_exists = os.path.isfile(filename)
            csv_file = open(filename, "w")
            if not file_exists:
                csv_file.write("run_name,file,idx,score,drawn,init,init_dist,irregular,jagged,failed,discontinuous,distance,complete,enclosed,v1v2angle,total,std,pos perc,area\n")

            for plane in planes:                
                dr_model=None
                if isinstance(plane, tuple):
                    plane_name, plane_dir = plane
                elif plane.startswith("grad_attrib"):
                    plane_name=plane
                    plane_dir, dr_model_name = plane.split("-")
                    if dr_models[dr_model_name]["type"] == "hybrid":
                        dr_model = HybridModel(**dr_models[dr_model_name])
                    elif dr_models[dr_model_name]["type"] == "dimensionality_reduction":
                        dr_model = GenericDRModel(**dr_models[dr_model_name])

                    plane_dir = [plane_dir for _ in range(n_samples)]
                    
                elif plane=="random":
                    plane_name="random"
                    plane_dir = ["random" for _ in range(n_samples)]
                    
                    
                run_name = f"{device}/{nids_model.name}_{nids_model.threshold:.3f}/{seed}_{start_name}_{plane_name}"

                boundary_path = f"../adversarial_data/{run_name}/"
                plot_path = f"exp_figs/db_vis/{run_name}/"
                log_path = f"boundary_logs/{run_name}"
                if not os.path.exists(boundary_path):
                    os.makedirs(boundary_path)
                if not os.path.exists(plot_path):
                    os.makedirs(plot_path)
                if not os.path.exists(log_path):
                    os.makedirs(log_path)

                draw_i = 0
                for draw_plots, fn, idx, start, p in tqdm(zip(draw, file_names, target_idx, start_samples, plane_dir)):
                    
                    logger_level = None
                    write=True

                    print(run_name, idx, nids_model.predict(
                        start[np.newaxis, :]))

                    statistics = boundary_traversal(
                        baseline_nids=nids_model, dr_model=dr_model,
                        output_boundary_path=boundary_path, start_position=start, plane=p,
                         run_name=run_name, max_step=0.1,
                        max_iter=2000, logger_level=logger_level,
                        eps=1e-3*nids_model.threshold, draw_plots=draw_plots, idx=idx, write=write)
                    
                    if write:
                        csv_file.write(
                            ",".join(list(map(str, [run_name,fn, idx]+list(statistics.values())))))
                        csv_file.write("\n")
                        csv_file.flush()
                    else:
                        print(statistics)

            csv_file.close()

            # draw_db(dr_model, files,
            #         heat_map=-1, file_name=f"{dr_model.name}_{sampling_method}", bidirectional=bidirectional)
