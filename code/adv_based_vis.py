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
import scipy 
sys.path.insert(1, '../../mtd_defence/code')
import train_mtd_am

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


def feature_to_2d(x_0, features, q, func, feature_range=None):
    if feature_range is None:
        diff_vector = (features-x_0)
    else:    
        diff_vector = (features-x_0)/feature_range
    result, residual, _, _ = np.linalg.lstsq(q, diff_vector.T, rcond=None)
    
    if feature_range is None:
        plane_out = x_0+np.einsum("ki,lk->il", result, q)
    else:
        plane_out = x_0+np.einsum("ki,lk,l->il", result, q,feature_range)

    plane_out = plane_out.reshape([-1, q.shape[0]])

    feature_diff = np.sum(np.abs(plane_out-features), axis=1)
    print(feature_diff.shape)
    print(f"max feature diff {np.max(feature_diff)}")
    print(f"max residual {np.max(residual)}")

    pred_val = func(plane_out)
    true_val = func(features)

    max_error = np.argmax(np.abs(pred_val-true_val))
    print(
        f"max error index {max_error} plane val {pred_val[max_error]} true val {true_val[max_error]}")
   
    return result, pred_val, true_val


def plot_decision_boundary_3p(func, q, plot_range=[[-1.5, 2.5, 200], [-2, 2.5, 150]], boundary_file=None, symbols=None,
                              adv_samples=None, threshold=1, file_name="test", feature_range=None):

    boundary_path = np.genfromtxt(boundary_file, delimiter=",")
    
    x_0 = boundary_path[np.newaxis, 0]
    
    # plot decision boundary
    boundary_result, pred_val, true_val = feature_to_2d(
        x_0, boundary_path, q, func, feature_range)

    fig = go.Figure(go.Scattergl(x=boundary_result[0], y=boundary_result[1], name="boundary", mode="markers",
                                 hovertext=[
        f"index: {i}, plane_as: {val}, true_as: {true_val[i]}" for i, val in enumerate(pred_val)],
        marker=dict(size=10,
                    color=true_val,
                    symbol=symbols,
                    line=dict(
                        color='MediumPurple',
                        width=2
                    )
                    )))
    # plot adversarial examples
    if adv_samples is not None:
        adv_result, pred_val, true_val = feature_to_2d(
            x_0, adv_samples, q, func,feature_range)
        fig.add_trace(go.Scattergl(x=adv_result[0], y=adv_result[1], name="adv samples", mode="markers",
                                   hovertext=[
                                       f"index: {i}, plane_as: {val}, true_as: {true_val[i]}" for i, val in enumerate(pred_val)],
                                   marker=dict(size=10,
                                               color=true_val,
                                               symbol="triangle-up",
                                               line=dict(
                                                   color='MediumPurple',
                                                   width=2
                                               )
                                               )))
        result = np.hstack([boundary_result, adv_result])
    else:
        result = boundary_result
    
    # plot contour
    if plot_range is None:
        
        dir1 = np.linspace(np.min(
            result[0])-np.ptp(result[0])*0.1, np.max(result[0])+np.ptp(result[0])*0.1, 100)
        dir2 = np.linspace(np.min(
            result[1])-np.ptp(result[1])*0.1, np.max(result[1])+np.ptp(result[1])*0.1, 120)
    else:
        dir1 = np.linspace(*plot_range[0])
        dir2 = np.linspace(*plot_range[1])

    xv, yv = np.meshgrid(dir1, dir2)

    coord_mat = np.dstack([xv, yv])

    if feature_range is None:
        input_val = x_0+np.einsum("ijk,lk->ijl", coord_mat, q)
    
    else:
        input_val = x_0+np.einsum("ijk,lk,l->ijl", coord_mat, q, feature_range)
    
    input_val = input_val.reshape([-1, q.shape[0]])

    f_val = func(input_val)
    f_val = f_val.reshape(xv.shape)
    fig.add_trace(
        go.Contour(
            z=f_val,
            x=dir1,  # horizontal axis
            y=dir2,  # vertical axis
            opacity=0.4,
            
            contours=dict(
                showlabels=True,  # show labels on contours
                labelfont=dict(  # label font properties
                    size=12,
                    color='white',
                ),
                start=0,
                end=np.max(f_val),
                size=(threshold)/5.
                
            )
        ))

    # reverse order
    fig.data = fig.data[::-1]

    fig.write_html(f"exp_figs/db_vis/{file_name}.html")


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
        return start+np.einsum("j,jk,k->jk", step, direction,feature_range)

def linear_search(direction, original, step_size, threshold, func, diff_threshold=1e-3, step_count=0, logger=None, 
                  feature_range=None, max_iter=50, check_both_dir=True, return_step=False):
    # ensure direction is unit vector
    logger.info("begin linear search")
    direction/=np.linalg.norm(direction)

    step = np.copy(step_size)
    count = np.zeros(step_size.shape)
    same_label_count = np.zeros(step_size.shape)
    
    prev_label, scores = func(original, True)

    closest_score = np.copy(scores)
    closest_step = np.zeros(step_size.shape)

    # check direction does make anomaly score closer
    if check_both_dir:
        search_x = take_step(original, step, direction, feature_range) 
        _, test_scores = func(search_x, True)
        logger.info(f"test score {test_scores}, test step {step}")

        # check if they are in the same direction
        step_size[np.sign(threshold-scores) != np.sign(test_scores-scores)] *= -1
        step = np.copy(step_size)

    logger.info(f"initial score {scores}, initial step size {step_size}")

    while np.any(np.abs(threshold - scores) > diff_threshold):
        search_x = take_step(original, step, direction, feature_range)
        label, scores = func(search_x, True)

        # # update closest score and step
        closer_idx = np.abs(
            closest_score - threshold) > np.abs(scores - threshold)
        closest_score[closer_idx] = scores[closer_idx]
        closest_step[closer_idx] = step[closer_idx]

        # check if same label
        same_label = (label == prev_label)

        same_label_count[same_label] += 1

        same_label_count[~same_label] = 0
        step_size[same_label_count == 5] *= 2
        
        same_label_count[same_label_count == 5 ] = 0

        logger.info(
            f"step {step}, scores at {count}: {scores}, same label: {same_label}")
        # if different label, move in the other direction and decrease step size
        step_size[~same_label] *= -0.5
        step += step_size

        count[~same_label] = 0

        # update previous label and score
        prev_label = label
        count += 1
        if np.all(count > max_iter):
            fail_idx = np.abs(threshold - scores) > diff_threshold

            closest_x = take_step(original, closest_step, direction, feature_range)
            _, closest_score = func(closest_x, True)

            logger.warning(
                f"linear search failed, closest score {closest_score[fail_idx]}")
            logger.info("-"*50)
            if return_step:
                return closest_x, closest_score, closest_step
            else:
                return closest_x, closest_score
    logger.info("-"*50)
    if return_step:
        return search_x, scores, step-step_size
    else:
        return search_x, scores


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
    dot_prod=np.einsum("ij,ki->kj",A,normal)

    normal/=np.linalg.norm(normal, axis=1)
    dot1=dot_prod[:,0]
    dot2=dot_prod[:,1]

    # if close to zero, return previous value
    if (np.abs(dot1) < 1e-9).any() and (np.abs(dot2) < 1e-9).any():
        if logger is not None:
            logger.info(f"boundary == tangent")
        return prev

    # elif (np.abs(dot1)<1e-9):
    #     return -A[None,:,0]
    # elif (np.abs(dot2)<1e-9):
    #     return -A[None,:,1]
    
    coef=np.vstack([np.full((normal.shape[0]),1), -dot1/dot2])

    direction=np.einsum("ij, jk->ki", A, coef)
    if logger is not None:
        logger.info(f"dot1 {dot1}, dot2 {dot2}")

    # normalise direction
    direction /= np.linalg.norm(direction)
    
    if np.abs(angle(prev, direction)-np.pi)<0.5:
        direction*=-1
    
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


def step_size_estimate(func, x_n, step_size, tangent_d, feature_range, logger=None):
    test_sample=take_step(x_n, step_size, tangent_d, feature_range)
    test_score=func(test_sample)
    d=np.abs(test_score-func(x_n))
    
    R=(step_size**2+d**2)/(2*d) 
    if logger is not None:
        logger.info(f"step size {step_size} d {d} estimated radius {R}")
    return R*0.018
    
    

def curvature(func, x, grad, v=None):
    def grad_func(x):
        return gradient_estimate(func, x, 1e-6)

    H = gradient_estimate(grad_func, x, 1e-6)
    grad_mag=np.linalg.norm(grad,axis=1)
    if v is None:    
        P=np.identity(grad.shape[0])-np.einsum("ij,ik->ijk", grad, grad)
        matrix=1/grad_mag * np.einsum("mij,mjk,mkl->mil", P, H, P)
        return np.linalg.eigvals(matrix).real

    else:
        curvature=np.einsum("ij, ijk, ik->i", v, H, v)/(np.einsum("ij,ij->i", v,v)*grad_mag)
        return curvature[:,np.newaxis]

def find_perpendicular_vector(directional_derivative, A, direction, logger=None):
    
    tangent=np.hstack([directional_derivative[None, :,1],-directional_derivative[None,:,0]])*direction
    
    tangent_d=np.einsum("ij,kj->ik", tangent, A)

    tangent_d/=np.linalg.norm(tangent_d, axis=1,keepdims=True)
    # if logger is not None:
    #     logger.info(f"prev tangent and tangent angle {np.degrees(angle(prev_tangent, tangent_d))}")
        
    # if np.abs(angle(prev, direction)-np.pi)<0.5:
    #     tangent_d*=-1
    return tangent_d

def boundary_traversal(baseline_nids, start_position, output_boundary_path,
                       batch_size=8, tangent_guide=None, dr_model=None,
                       sampling_method=None, end_position=None, anchor=None, run_name="", init_search="binary", max_iter=500,
                       bidirectional=True, grad_est="fd", logger_level=logging.WARNING, eps=1e-3, step_size=1e-2, early_stopping=False,
                       draw_plots=True):

    # set up logger
    threshold = baseline_nids.threshold
    nids_name = baseline_nids.name

    logger = setup_logger(f'{nids_name}_{run_name}_{threshold:.3f}',
                          f'boundary_logs/{nids_name}_{run_name}_{threshold:.3f}.log', logger_level)

    feature_range = (scaler.data_max_ - scaler.data_min_)
    
    # if start_position can either be a filename of data or a single data point
    if isinstance(start_position, str):
        dataset = get_dataset(start_position, batch_size, frac=1, shuffle=False,
                              read_with="pd", seed=42, dtype="float32", skip_header=False)
    else:
        dataset=start_position
        if dataset.ndim==1:
            dataset = start_position[np.newaxis,:]
        

    if grad_est == "mc":
        def grad_func(f, x, delta_t=1e-5):
            return monte_carlo_estimate(
                f, x, 4096, delta_t, logger=logger)
    elif grad_est == "fd":
        def grad_func(f, x, delta_t=1e-5, direction=None, feature_range=None):
           
            return gradient_estimate(f, x, delta_t, direction, feature_range)
           
    else:
        raise Exception("invalid gradient estimate")

   
    count = 0

    for x_start in tqdm(dataset):
        output_bd=f"{output_boundary_path}{count}.csv"
        logger.info(f"traversing {count}")
        logger.info("="*50)
        
        if draw_plots:
            # plots and data for statistics
            fig, axs = plt.subplots(2, 2, figsize=(12, 6))
            dist_ben = []
            dist_prev = []
            boundary_gradient_angles = []
            relationship = []
            profiles = []
            ord = 2
        with open(output_bd, "w") as adv_f:


            if not isinstance(x_start, np.ndarray):
                x_start = x_start.numpy()
            if x_start.ndim == 1:
                x_start = np.expand_dims(x_start, axis=0)

            start_label, start_score = baseline_nids.decision(x_start, True)
            logger.info(f"x start score: {start_score}, label: {start_label}")

            np.savetxt(adv_f, x_start, delimiter=',')

            # find initial bondary sample
            if end_position is None:
                if init_search != "linear":
                    raise ValueError(
                        "if end position is not provided then init_search must be linear")
            else:
                x_end = np.tile(end_position, [x_start.shape[0], 1])
                end_label, end_score = baseline_nids.decision(x_end, True)
                logger.info(f"x end score: {end_score}, label: {end_label}")

            if init_search == "binary":
                if start_label == end_label:
                    raise ValueError(
                        "start and end have same label which does not work with binary search")

                x_n, x_n_score = binary_search(
                    x_end, x_start, eps, baseline_nids.decision, logger=logger)

            elif init_search == "linear":
                x_n_score=0
                x_n=np.copy(x_start)
                while np.abs(x_n_score-threshold)>eps:
                    # find the search direction
                    if end_position is None: 
                        logger.info("search for boundary point with gradient")
                        init_search_direction=grad_func(baseline_nids.predict,
                                    x_start, delta_t=step_size) * np.sign(threshold-start_score)
                    else:
                        logger.info("search for boundary point with end sample")
                        if feature_range is None:
                            init_search_direction = (x_end-x_start)
                        else:
                            init_search_direction = (x_end-x_start)/feature_range

                    x_n, x_n_score = linear_search(
                        init_search_direction, x_n, np.full(
                            start_score.shape, 1e-1), threshold, 
                        baseline_nids.decision, diff_threshold=eps, logger=logger, feature_range=feature_range, max_iter=20)
                
                if end_position is None:
                    x_end=x_n
                
                if np.any(np.abs(threshold - x_n_score) > eps):
                    logger.warning(
                        f"initial linear search did not find boundary score, final score is {x_n_score}")
            else:
                raise Exception("unknown init_search")

            logger.info(f"initial boundary score: {x_n_score}")
            x_init=np.copy(x_n)
            np.savetxt(adv_f, x_n, delimiter=',')

            # find guiding direction for tangent
            if tangent_guide == "anchor":
                if anchor is None:
                    guiding_dir = rng.uniform(scaler.data_min_, scaler.data_max_, x_start.shape)
                else:
                    guiding_dir = (anchor-x_start)

            elif tangent_guide.endswith("grad_attrib"):
                # find feature attributions of start to x_n

                attributions, _ = integrated_gradients(
                    dr_model.transform, x_start, x_n, m_steps=128, recon=True)

                if tangent_guide.startswith("perp"):
                    # find feature attributions of start to x_n that is perpendicular to x_n-x_start
                    x_n_lat = dr_model.transform(x_n)
                    x_start_lat = dr_model.transform(x_start)

                    lat_dir = x_n_lat - x_start_lat

                    perp_dir = [lat_dir[0, 1], -lat_dir[0, 0]]

                    guiding_dir = perp_dir[0] * attributions[0, 0, :] + \
                        perp_dir[1] * attributions[0, 1, :]
                else:
                    guiding_dir = np.sum(attributions, axis=1)
            else:
                raise Exception("No tangent guide type found")

            
            boundary_direction = (x_n-x_start)

            if feature_range is not None:
                guiding_dir/=feature_range
                boundary_direction/=feature_range
                
            tangent_d = guiding_dir
            direction = 1
            step_size = np.full(start_score.shape, step_size)
            corrected_count = [0, 0]
            corrected_distance=0
            symbols = ["star", "triangle-up"]
            
            A = np.hstack([guiding_dir.T, boundary_direction.T])
            A, _ = np.linalg.qr(A)
            early_stop_flag=False
            logger.info("*"*50)
            for t in tqdm(range(max_iter)):
                if bidirectional and (t == max_iter//2 or early_stop_flag):
                    if direction==-1:
                        break
                    direction = -1
                    x_n = x_init
                    tangent_d=-guiding_dir
                    early_stop_flag=False
                    

                # grad_f = grad_func(baseline_nids.predict, x_n, delta_t=1e-4)
                
                # if feature_range is not None:
                #     grad_f*=feature_range

                # tangent_d = intersection_vector(
                #     A, grad_f, tangent_d, logger=logger)
                
                grad_f = grad_func(baseline_nids.predict,
                                       x_n, delta_t=1e-5, direction=A.T, feature_range=feature_range)
                logger.info(f"gradient at x_n {grad_f/np.linalg.norm(grad_f,axis=1,keepdims=True)}")
                
                tangent_d=find_perpendicular_vector(grad_f, A, direction, logger=logger)
                
                # find tangent direction in boundary plane
                res, residual, _, _ = np.linalg.lstsq(A, tangent_d.T, rcond=None)
                res /= np.linalg.norm(res, axis=0)
                
                logger.info(f"tangent direction: {res.T}, residual: {residual}")
                
                grad_f=np.einsum("ij,kj->ik", grad_f, A)/feature_range
                
                # curvature calculation
                step_size=step_size_estimate(baseline_nids.predict, x_n, np.array([1e-3]), tangent_d, feature_range, logger=logger)
                step_size=np.minimum(step_size, 0.2)
                
                # logger.info(f"step size est {step_size}")
                # curvature_profile = curvature(
                #     baseline_nids.predict, x_n, grad_f)
                # profiles.append(curvature_profile)
                # logger.info(f"maximum curvature {np.max(curvature_profile)}, minimum curvature {np.min(curvature_profile)}")
                # step_size=0.173/np.abs(curvature)
                # step_size=np.array([1e-1])
                # step_size=np.clip(step_size, 0.001, 0.1)
                
                # take a step and get score
                x_np1 = take_step(x_n, step_size, tangent_d, feature_range)
                x_np1_score = baseline_nids.predict(x_np1)
                
                logger.info(f"after step {t} score {x_np1_score}")
                
                # find coordinate in boundary plane
                if feature_range is not None:
                    coord=(x_np1-x_start)/feature_range
                else:
                    coord=(x_np1-x_start)
                x_np1_coord, coord_residual, _, _ = np.linalg.lstsq(A, coord.T, rcond=None)
                logger.info(f"x_np1 coord {x_np1_coord.T} residual {coord_residual}")

                # find change along gradient
                gradient_score = baseline_nids.predict(take_step(x_n, step_size, grad_f/np.linalg.norm(grad_f, axis=1), feature_range))
                
                # original score
                x_n_score = baseline_nids.predict(x_n)

                # save uncorrected
                # np.savetxt(adv_f, x_np1, delimiter=',')
                # symbols.append("diamond")

                # find angle between boundary and gradient
                if draw_plots:
                    bg = angle(grad_f, A.T)
                    boundary_gradient_angles.append(bg)
                
                
                
                logger.info(
                    f"gradient_score diff {gradient_score-x_n_score} tangent_score diff {x_np1_score-x_n_score}")
                if np.abs(gradient_score-x_n_score)<np.abs(x_np1_score-x_n_score):
                    logger.info("tangent > gradient")

                # correction step
                correction_count = 0
                # correction_step = np.clip((threshold - end_score) * 10,
                #                           -0.02, 0.02)
                correction_step = np.full(
                    (x_np1_score.shape[0],), np.sign(threshold-x_np1_score)* step_size/20.)

                while np.any(np.abs(threshold - x_np1_score) > eps):
                    update_idx = np.abs(threshold - x_np1_score) > eps

                    grad_f = grad_func(baseline_nids.predict,
                                       x_np1[update_idx], delta_t=1e-5, direction=A.T, feature_range=feature_range)
                    
                    plane_grad=np.einsum("ij,kj->ik", grad_f, A)
                    
                    res, residual, _, _ = np.linalg.lstsq(A, plane_grad.T, rcond=None)
                    res /= np.linalg.norm(res, axis=0)
                    
                    logger.info(f"correction direction {grad_f/np.linalg.norm(grad_f, axis=1,keepdims=True)}, res {res.T}, residual {residual}")

                    points, scores, corrected_distance = linear_search(
                        plane_grad, x_np1[update_idx], correction_step[update_idx], threshold,
                        baseline_nids.decision, diff_threshold=eps, step_count=correction_count, 
                        logger=logger, feature_range=feature_range, max_iter=20, check_both_dir=False, return_step=True)
                    
                    

                    x_np1[update_idx] = points
                    x_np1_score[update_idx] = scores
                    correction_count += 1

                    logger.info(
                        f"correction step {correction_count}, end_score {x_np1_score}, corrected_step {corrected_distance}")

                    if correction_count == 10:
                        logger.warning(
                            f"correction failed at {t}, end_score {x_np1_score}, correction_step {correction_step}")
                        break
                    
                if feature_range is not None:
                    coord=(x_np1-x_start)/feature_range
                else:
                    coord=(x_np1-x_start)
                x_np1_coord, coord_residual, _, _ = np.linalg.lstsq(A, coord.T, rcond=None)
                logger.info(f"corrected coord {x_np1_coord.T} residual {coord_residual}")
                
                if correction_count == 0:
                    symbols.append("circle")
                elif correction_count > 1:
                    symbols.append("cross")
                    corrected_count[1] += 1
                    logger.warning(f"correction count is over 2")
                else:
                    symbols.append("x")
                    corrected_count[0] += 1

                logger.info("*"*50)
                np.savetxt(adv_f, x_np1, delimiter=',')

                if sampling_method == "linear":
                    # linearly sample
                    lin_samples = linear_sample(x_start, x_np1_score, 10)
                    np.savetxt(adv_f, lin_samples, delimiter=',')

                if sampling_method == "neighbour":
                    # neighbourhood sample
                    attributions, _, _ = integrated_gradients(
                        dr_model.transform, x_start, x_np1_score, m_steps=128, recon=False, reduce_points=True)
                    attributions = attributions.numpy()

                    neig_samples = sample_near_point(
                        x_np1_score, attributions[np.newaxis, 0, 0,
                                                  :], attributions[np.newaxis, 0, 1, :], feature_range,
                        range=[0.3, 0.3], steps=[3, 3])
                    np.savetxt(adv_f, neig_samples, delimiter=',')

                x_n = np.copy(x_np1)
                if draw_plots:
                    distance = ln_distance(scaler.transform(
                        x_start), scaler.transform(x_np1), ord)
                    dist_ben.append(distance)
                    distance2 = ln_distance(scaler.transform(
                        x_n), scaler.transform(x_np1), ord)
                    dist_prev.append(distance2)
                
                
                if early_stopping:
                    theta=angle((x_init-x_start)/feature_range, (x_n-x_start)/feature_range)
                    logger.info(f"theta {np.degrees(theta)}")
                    #check if angle is close to pi
                    if bidirectional:
                        if np.abs(theta-np.pi)<1e-1:
                            early_stop_flag=True 
                            logger.info("bidirectional early stopping")
                    else:
                        # check if it passes half way
                        if np.abs(theta-np.pi)<1e-1:
                            early_stop_flag=True
                            
                            logger.info("unidirectional half way")
                        # if it passes and returns back to 0 we stop
                        if early_stop_flag and np.abs(theta)<1e-1:
                            
                            logger.info("unidirectional early stopping")
                            break 

            print(f"number of corrected points: {corrected_count[0]}")
            print(f"number of disjoint points {corrected_count[1]}")
            
            
            if draw_plots:
                dist_ben = np.squeeze(dist_ben)
                dist_prev = np.squeeze(dist_prev)
                # if bidirectional:
                #     mid=max_iter//2
                # else:
                #     mid=max_iter
                # if dist_ben.ndim == 1:
                #     axs[0].plot(np.hstack((dist_ben[mid:][::-1], dist_ben[:mid])))
                # else:
                #     axs[0].plot(
                #         np.vstack((dist_ben[mid:][::-1, :], dist_ben[:mid])))
                axs[0][0].plot(dist_ben)
                axs[0][0].set_title(
                    f"distance from benign sample in l{ord:.2f} norm", wrap=True)

                axs[0][1].plot(dist_prev)
                axs[0][1].set_title(
                    f"distance from previous point in l{ord:.2f} norm", wrap=True)

                axs[1][0].plot(np.degrees(boundary_gradient_angles))
                axs[1][0].set_title(f"angle between boundary and gradient", wrap=True)

                relationship=np.array(relationship)
                axs[1][1].plot(relationship)
                axs[1][1].set_title(
                    f"angle between guiding dir and gradient", wrap=True)

                # transform profiles into acceptable format by joypy
                # profiles = np.vstack(profiles)
                # # group n rows
                # n = 10
                # profiles = profiles.reshape(-1, n, profiles.shape[1])

                # ridgeline = go.Figure()
                # for data_line in profiles:
                #     ridgeline.add_trace(go.Violin(x=data_line.flatten()))

                # ridgeline.update_traces(orientation='h', side='positive', width=5, points=False)
                # ridgeline.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)

                # ridgeline.write_html(f"exp_figs/meta_plots/{nids_name}_{run_name}_ridge.html")

                fig.tight_layout()
                fig.savefig(f"exp_figs/meta_plots/{nids_name}_{run_name}_{count}.png")
                
                
                np.savetxt(adv_f, x_end, delimiter=',')
                symbols.append("star")

                plot_decision_boundary_3p(baseline_nids.predict, A, plot_range=None, boundary_file=output_bd,
                                    symbols=symbols, adv_samples=anchor, threshold=threshold,feature_range=feature_range,
                                    file_name=f"{nids_name}_{threshold:.3f}_{init_search}_{tangent_guide}_{run_name}_{count}")
            
            count += 1


def ln_distance(x1, x2, ord):
    return np.sum(np.abs(x1-x2)**ord, axis=1, keepdims=True)**(1./ord)


def angle(v1, v2):
    v1 /= np.linalg.norm(v1, axis=1, keepdims=True)
    v2 /= np.linalg.norm(v2, axis=1, keepdims=True)
    return np.squeeze(np.arccos(np.clip(np.einsum('ij,kj->ik', v1, v2), -1.0, 1.0)))


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


def read_adversarial_file(log_file, adv_csv, adv_ori_csv, out_path, idx_offset, target_adv_idx):

    count = 0
    with open(log_file, "r") as f:
        record = False
        c = 0
        for line in f.readlines():

            if line.startswith("original"):
                record = True
                continue
            if line.startswith("mutation"):
                record = False

            if record:

                if count == target_adv_idx:
                    line = line.rstrip().split(",")
                    num_craft_pkt = int(float(line[6].split(" ")[1]))
                    adv_start_idx = int(line[3]) - \
                        idx_offset - num_craft_pkt + 1
                    mal_idx = int(line[2])
                    break

                count += 1
            else:
                continue

    adv_features = pd.read_csv(
        adv_csv, skiprows=adv_start_idx, nrows=num_craft_pkt + 1, header=None)
    ori_adv_features = pd.read_csv(
        adv_ori_csv, skiprows=mal_idx + 1, nrows=1, header=None, usecols=list(range(100)))

    result = pd.concat([ori_adv_features, adv_features])
    with open(out_path, "w") as adv_f:
        np.savetxt(adv_f, result.to_numpy(), delimiter=',')
    return result.to_numpy("float32")


def get_benign_sample(benign_path, idx):
    benign_sample = pd.read_csv(benign_path, usecols=list(range(100)),skiprows=idx, nrows=1, header=None)
    return benign_sample.to_numpy()

def sample_n_from_csv(filename, n, total_rows=None,seed=42):
    rng=np.random.default_rng(seed)
    if total_rows is None:
        with open(filename,"r") as fh:
            total_rows = sum(1 for row in fh)
    if(n>total_rows):
        raise Exception("n cannot be larger than total rows") 
    skip_rows = rng.choice(total_rows, total_rows-n-1, replace=False)
    return pd.read_csv(filename, usecols=list(range(100)), skiprows=skip_rows, header=1).to_numpy()
      


def get_closest_benign_sample(benign_path, adv_example, transform_func=None):
    """finds the closest benign sample to malicious sample

    Args:
        benign_path (string): path to benign samples
        adv_example (ndarray): adversarial samples with format malicious->craft->adversarial
        transform_func (function, optional): dimensionality reduction function. if supplied it will find the closest benign sample
        in the latent space, else it finds the closet benign sample in feature space. Defaults to None.

    Returns:
        np.ndarray: the closest benign sample
    """
    traffic_ds = get_dataset(benign_path, 1024,
                             scaler=None, frac=1, read_with="tf", dtype="float32",
                             seed=42, skip_header=True, shuffle=True)
    # first one is malicious
    adv_example = adv_example[1:]

    if transform_func is not None:
        adv_example = transform_func(adv_example)

    closest_dist = None
    nearest_sample = None

    total = 0
    for data in tqdm(traffic_ds):
        data = data.numpy()
        if transform_func is not None:
            latent = transform_func(data)
        else:
            latent = data

        total += data.shape[0]
        distance = np.linalg.norm(
            latent[:,  None, :] - adv_example[None, :, :], axis=-1)
        
        closest_idx = np.argmin(distance, axis=0,keepdims=True)
        
        current_closest_dist = np.take_along_axis(distance, closest_idx, axis=0)
        
        current_nearest_sample = latent[closest_idx[0]]
        
        
        
        if closest_dist is None:
            closest_dist = current_closest_dist
            nearest_sample = current_nearest_sample
        else:
            
            update_idx = (current_closest_dist < closest_dist)
            
            closest_dist[update_idx] = current_closest_dist[update_idx]
            
            
            nearest_sample[update_idx[0]] = current_nearest_sample[update_idx[0]]
            
    

    print("processed", total)
    return nearest_sample


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


if __name__ == '__main__':
    dataset = "uq"
    mtd_model_path = f"../../mtd_defence/models/{dataset}/mtd/Cam_1/fm0_mm1_am20"
    scaler_type = "min_max"
    scaler_path = f"../../mtd_defence/models/uq/autoencoder/Cam_1_scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    target_nids = {
        # "model3_SOM": {"path": f"{mtd_model_path}/model3.pkl", "thresholds": [2.4904096433417897]},
        # "model2_OCSVM": {"path": f"{mtd_model_path}/model2.pkl", "threshold": 3.3806848104628235},
        # "model15_SOM": {"path": f"{mtd_model_path}/model15.pkl", "thresholds": [2.841178429133494]},
        # "model12_OCSVM": {"path": f"{mtd_model_path}/model12.pkl", "thresholds": [2.242774499012671]},
        # "baseline_kitsune": {"path": f"../../mtd_defence/models/{dataset}/kitsune/Cam_1.pkl",
        #                      "thresholds": np.linspace(0.1,0.28151878818499115, 10)},
        "baseline_kitsune": {"thresholds": [0.28151878818499115], "params": {
                             "path": f"../../mtd_defence/models/{dataset}/kitsune/Cam_1.pkl",
                             "func_name": "process",
                             "scale_output": 1,
                             "save_type": "pkl"}}
    }

    batch_size = 10
    sampling_method = None
    distinguish_start_end = sampling_method == None
    adv_atk = "kitsune_0.5_10_3_False_pso0.5"

    benign_path = "../../mtd_defence/datasets/uq/benign/Cam_1.csv"
    atk_type = "bt"
    # adv_path = f"../../mtd_defence/datasets/uq_network/adversarial/decision_boundary_adversarial/db_vis_0.01_20_3_False_pso0.5_None/csv/decision_boundary_adversarial_iter_0.csv"
    # ae_name = "min_max_mean_10.0_recon_loss_sw_loss_contractive_loss_adam_denoising_2d_0.001_double_recon"

    dr_model_name = "pca"
    dr_models = {"pca": {"type": "hybrid", "name": "pca", "save_type": "pkl", "path": "../models/pca.pkl",
                         "func_name": "transform", "scaler": scaler, "threshold": 0.3},
                 "umap": {"type": "hybrid", "name": "umap", "save_type": "pkl", "path": "../models/umap.pkl",
                          "func_name": "transform", "scaler": scaler, "threshold": 0.3},
                 "ae": {"type": "hybrid", "name": "ranking_loss_dist_loss", "save_type": "tf", "path": "../models/min_max_mean_10.0_ranking_loss_dist_loss_adam_denoising_2d_0.001",
                        "func_name": "call", "dr_output_index": 0, "ad_output_index": 1, "dtype": "float32", "threshold": 0.3},
                 "lle": {"type": "dimensionality_reduction", "name": "lle", "save_type": "pkl", "path": "../models/lle.pkl",
                         "func_name": "transform", "scaler": scaler, },
                 }

    # dr_name = ""

    if dr_models[dr_model_name]["type"] == "hybrid":
        dr_model = HybridModel(**dr_models[dr_model_name])
    elif dr_models[dr_model_name]["type"] == "dimensionality_reduction":
        dr_model = GenericDRModel(**dr_models[dr_model_name])

    target_adv_idx = [10]

    bidirectional = True

    default_nids = GenericADModel("baseline_kitsune",  **{
        "path": f"../../mtd_defence/models/{dataset}/kitsune/Cam_1.pkl",
        "func_name": "process", "threshold": 0.28151878818499115,
        "save_type": "pkl"})

    files = {
        "benign": {"file_path": benign_path, "color_scale": "Picnic", "frac": 0.05, "batch_size": 1024,
                   "plot_batch_size": 1, "draw_type": "background", "symbol": "circle", "shuffle": True,
                   "opacity": 0.04, "skip_header": True,
                   "nids_model": default_nids,
                   },
        # "ACK": {"file_path": "../../mtd_defence/datasets/uq/malicious/Cam_1/Cam_1_ACK_Flooding.csv", "color_scale": "Picnic", "frac": 0.03, "batch_size": 1024, "plot_batch_size": 1, "skip_header": True},
        # "SYN": {"file_path": "../../mtd_defence/datasets/uq/malicious/Cam_1/Cam_1_SYN_Flooding.csv", "color_scale": "Picnic", "frac": 0.05, "batch_size": 1024, "plot_batch_size": 1, "skip_header": True},
        # "UDP": {"file_path": "../../mtd_defence/datasets/uq/malicious/Cam_1/Cam_1_UDP_Flooding.csv", "color_scale": "Picnic", "frac": 0.02, "batch_size": 1024, "plot_batch_size": 1, "skip_header": True},
        # "ACK_adv": {"file_path": "../../mtd_defence/datasets/uq/adversarial/Cam_1/ACK_Flooding/autoencoder_0.5_10_3_False_pso0.5/csv/Cam_1_ACK_Flooding_iter_0.csv", "color_scale": "Picnic", "frac": 0.1, "batch_size": 1024, "plot_batch_size": 1},
        # "SYN_adv": {"file_path": "../../mtd_defence/datasets/uq/adversarial/Cam_1/SYN_Flooding/autoencoder_0.5_10_3_False_pso0.5/csv/Cam_1_SYN_Flooding_iter_0.csv", "color_scale": "Picnic", "frac": 0.1, "batch_size": 1024, "plot_batch_size": 1},
        # "UDP_adv": {"file_path": "../../mtd_defence/datasets/uq/adversarial/Cam_1/UDP_Flooding/autoencoder_0.5_10_3_False_pso0.5/csv/Cam_1_UDP_Flooding_iter_0.csv", "color_scale": "Picnic", "frac": 0.1, "batch_size": 1024, "plot_batch_size": 1},
    }
    # create_adversarial_examples(
    #     baseline_kitsune, scaler_path, benign_path, adv_path, atk_type, threshold)
    # random_sample(
    #     baseline_kitsune, scaler_path, benign_path, adv_path, threshold)
    # modified_hsj(
    #     baseline_kitsune, scaler_path, benign_path, adv_path, threshold, search_only=False)
    for i in target_adv_idx:
        # adv_path = f"../adversarial_data/Cam_1_adv_{i}.csv"
        # files[f"adv_{i}"] = {"file_path": adv_path, "color_scale": "Bluered", "frac": 1, "batch_size": 1, "shuffle": False,
        #                      "plot_batch_size": 1, "skip_header": False, "draw_type": "adv_sample", "opacity": 1,
        #                      "start_color": "red", "end_color": "goldenrod", "symbol": "star-triangle-up",
        #                      "nids_model": default_nids}

        # adv_samples=mal_sample+craft_sample+adv_sample
        # adv_samples = read_adversarial_file(f"../../mtd_defence/datasets/{dataset}/adversarial/Cam_1/ACK_Flooding/{adv_atk}/logs/Cam_1_ACK_Flooding_iter_0.txt",
        #                                     f"../../mtd_defence/datasets/{dataset}/adversarial/Cam_1/ACK_Flooding/{adv_atk}/csv/Cam_1_ACK_Flooding_iter_0.csv",
        #                                     f"../../mtd_defence/datasets/{dataset}/malicious/Cam_1/Cam_1_ACK_Flooding.csv",
        #                                     adv_path, 854685, i)
        
        # benign_sample = get_closest_benign_sample(
        #     benign_path, adv_samples, scaler.transform)

        # benign_sample=benign_sample[None,-1]
        # adversarial_sample = adv_samples[None, -1]
        # mal_sample = adv_samples[None, 0]
        seed=42
        benign_sample=sample_n_from_csv(benign_path, 1000, total_rows=854685, seed=seed)
        
        
        # bt_config={"start":("adv", adversarial_sample),"end": ("mal", mal_sample), "anchor":("ben",benign_sample)}
        bt_config={"start":("ben",benign_sample), "end": ("none", None), "anchor":("none",None)}
        
        # benign_sample=get_benign_sample(benign_path, i*100+1000)


        for nids_name, config in target_nids.items():

            for t in config["thresholds"]:
                # scale threshold
                t *= config["params"]["scale_output"]
                nids_model = GenericADModel(
                    nids_name, threshold=t, **config["params"])

                print(f"traversing boundary for {nids_name} at threshold {t}")
                run_name=f"{seed}_{bt_config['start'][0]}_{bt_config['end'][0]}_{bt_config['anchor'][0]}"
                boundary_path = f"../adversarial_data/Cam_1_{nids_name}_{atk_type}_{run_name}_{t:.3f}/"
                if not os.path.exists(boundary_path):
                    os.mkdir(boundary_path)
                
                files[f"{nids_name}_boundary_{i}_{t:.3f}"] = {"file_path": boundary_path, "color_scale": "balance", "shuffle": False,
                                                              "frac": 1, "batch_size": 1, "plot_batch_size": benign_sample.shape[0],
                                                              "skip_header": False, "draw_type": "boundary",
                                                              "nids_model": nids_model,
                                                              "opacity": 0.8, "start_color": "blue", "end_color": "red", "symbol": "star-dot"}

                boundary_traversal(
                    baseline_nids=nids_model, dr_model=dr_model,
                    output_boundary_path=boundary_path, start_position=bt_config["start"][1], batch_size=1,
                    tangent_guide="anchor", sampling_method=sampling_method, end_position=bt_config["end"][1], anchor=bt_config["anchor"][1], run_name=run_name,
                    init_search="linear", max_iter=1000, bidirectional=bidirectional, grad_est="fd", logger_level=logging.INFO,
                    eps=1e-6*config["params"]["scale_output"], step_size=1e-2, early_stopping=True, draw_plots=False)

    # draw_db(dr_model, files,
    #         heat_map=-1, file_name=f"{dr_model.name}_{sampling_method}", bidirectional=bidirectional)
