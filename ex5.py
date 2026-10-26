#  implementation for exercise 5 in Bayesian Learning course.
import numpy as np
from matplotlib import pyplot as plt
from ex5_utils import load_im_data, BayesianLinearRegression, gaussian_basis_functions, accuracy, Gaussian, plot_ims


def get_posterior_params(data_samples, prior_mu, prior_var, noise_var):
    num_samples = data_samples.shape[0]

    # precision is inverse of variance
    data_precision = 1 / noise_var
    prior_precision = 1 / prior_var

    # posterior precision
    posterior_precision = prior_precision + (num_samples * data_precision)

    # posterior variance
    posterior_var = 1 / posterior_precision

    # posterior mean
    sum_data = np.sum(data_samples, axis=0)
    posterior_mu = (prior_mu * prior_precision + data_precision * sum_data)/posterior_precision

    return posterior_mu, posterior_var

def get_decision_boundary(mu_plus, mu_minus):
    # vector difference between the two means
    diff_mu = mu_plus - mu_minus

    # squared norms
    sorm_sq_plus = np.sum(mu_plus ** 2)
    sorm_sq_minus = np.sum(mu_minus ** 2)

    # avoid division by zero if means are horizontally aligned (in the same plane)
    if np.isclose(diff_mu[1], 0):
        return None, None

    # calculate components from on Eq 1.6
    # y = ( (|mu+|^2 - |mu-|^2) / 2(diff_y) ) - ( diff_x / diff_y ) * x

    intercept = (sorm_sq_plus - sorm_sq_minus) / (2 * diff_mu[1])
    slope = -diff_mu[0] / diff_mu[1]

    return slope, intercept




def main():
    # ------------------------------------------------------ section 1
    # define question variables
    sig, sig_0 = 0.1, 0.25
    mu_p, mu_m = np.array([1, 1]), np.array([-1, -1])

    # sample 5 points from each class
    np.random.seed(0)
    x_p = np.array([.5, 0])[None, :] + np.sqrt(sig) * np.random.randn(5, 2)
    x_m = np.array([-.5, -.5])[None, :] + np.sqrt(sig) * np.random.randn(5, 2)

    # <your code here>

    # calculte MMSE parameters --> posterior mean and covariance
    post_mu_p, post_var_p = get_posterior_params(x_p, mu_p, sig_0, sig)
    post_mu_m, post_var_m = get_posterior_params(x_m, mu_m, sig_0, sig)

    plt.figure(figsize=(8, 8))

    # plot data points
    plt.scatter(x_p[:, 0], x_p[:, 1], label='Class +', c='blue')
    plt.scatter(x_m[:, 0], x_m[:, 1], label='Class -', c='red')

    x_domain = np.linspace(-3, 3, 100)

    # plot MMSE decision boundary
    slope_mmse, intercept_mmse = get_decision_boundary(post_mu_p, post_mu_m)
    plt.plot(x_domain, slope_mmse * x_domain + intercept_mmse, 'k-', linewidth=2.5, label='Mean Boundary')


    # sample 10 decision boundaries from the posterior distributions
    np.random.seed(1)
    for i in range (10):
        # sample means from the posterior distributions N(mu_N, Sigma_N * I)
        sample_mu_p = np.random.multivariate_normal(post_mu_p, post_var_p * np.eye(2))
        sample_mu_m = np.random.multivariate_normal(post_mu_m, post_var_m * np.eye(2))

        slope_sample, intercept_sample = get_decision_boundary(sample_mu_p, sample_mu_m)
        if i == 0:
            plt.plot(x_domain, slope_sample * x_domain + intercept_sample, alpha=0.5, label='Sampled Boundaries')
        else:
            plt.plot(x_domain, slope_sample * x_domain + intercept_sample, alpha=0.5)

    plt.title('Decision Boundaries')
    plt.legend()
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.grid(True, alpha=0.5)
    plt.show()

    # ------------------------------------------------------ section 2
    # load image data
    (dogs, dogs_t), (frogs, frogs_t) = load_im_data()

    # split into train and test sets
    train = np.concatenate([dogs, frogs], axis=0)
    labels = np.concatenate([np.ones(dogs.shape[0]), -np.ones(frogs.shape[0])])
    test = np.concatenate([dogs_t, frogs_t], axis=0)
    labels_t = np.concatenate([np.ones(dogs_t.shape[0]), -np.ones(frogs_t.shape[0])])

    # ------------------------------------------------------ section 2.1
    nus = [0, 1, 5, 10, 25, 50, 75, 100]
    train_score, test_score = np.zeros(len(nus)), np.zeros(len(nus))
    for i, nu in enumerate(nus):
        beta = .05 * nu
        print(f'QDA with nu={nu}', end='', flush=True)

        # <your code here>

        # fit Gaussian models to each class
        g_dogs = Gaussian(beta=beta, nu=nu).fit(dogs)
        g_frogs = Gaussian(beta=beta, nu=nu).fit(frogs)

        # define the prediction function log --> P(x|dog) - log P(x|frog)
        # if >0 predict dog(+1), else frog(-1)
        def qda_predict(X):
            log_likelihood_dogs = g_dogs.log_likelihood(X)
            log_likelihood_frogs = g_frogs.log_likelihood(X)
            return np.sign(log_likelihood_dogs - log_likelihood_frogs)

        # calculate accuracy
        train_predictions = qda_predict(train)
        test_predictions = qda_predict(test)

        train_score[i] = accuracy(train_predictions, labels)
        test_score[i] = accuracy(test_predictions, labels_t)

        print(f': train={train_score[i]:.2f}, test={test_score[i]:.2f}', flush=True)

    plt.figure()
    plt.plot(nus, train_score, lw=2, label='train')
    plt.plot(nus, test_score, lw=2, label='test')
    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel(r'value of $\nu$')
    plt.show()

    # ------------------------------------------------------ section 2.2
    # define question variables
    beta = .02
    sigma = .1
    Ms = [250, 500, 750, 1000, 2000, 3000, 5750]
    train_score, test_score = np.zeros(len(Ms)), np.zeros(len(Ms))

    blr = None
    for i, M in enumerate(Ms):
        print(f'Gaussian basis functions using {M} samples', end='', flush=True)

        # < your code here >

        # select M random centers from each class (dogs and frogs)
        # if M > len(dogs), use all samples
        current_M = min(M, len(dogs))

        # concatinate first M dogs and frogs to create centers
        centers = np.concatenate([dogs[:current_M], frogs[:current_M]], axis=0)

        # initialize Bayesian LR model with Gaussian basis functions
        basis_functions = gaussian_basis_functions(centers, beta=beta)

        # initialize BLR and fit the model
        # prior mean = 0(vector), prior COV = I
        num_centers = centers.shape[0]
        theta_mean = np.zeros(num_centers)
        theta_cov = np.eye(num_centers)

        blr = BayesianLinearRegression(theta_mean, theta_cov, sigma, basis_functions)
        blr.fit(train, labels)

        # predict and score the model on train and test sets
        train_predictions = np.sign(blr.predict(train))
        test_predictions = np.sign(blr.predict(test))

        train_score[i] = accuracy(train_predictions, labels)
        test_score[i] = accuracy(test_predictions, labels_t)


        print(f': train={train_score[i]:.2f}, test={test_score[i]:.2f}', flush=True)

    plt.figure()
    plt.plot(Ms, train_score, lw=2, label='train')
    plt.plot(Ms, test_score, lw=2, label='test')
    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('# of samples')
    plt.xscale('log')
    plt.show()

    # calculate how certain the model is about the predictions
    d = np.abs(blr.predict(dogs_t) / blr.predict_std(dogs_t))
    inds = np.argsort(d)
    # plot most and least confident points
    plot_ims(dogs_t[inds][:25], 'least confident')
    plot_ims(dogs_t[inds][-25:], 'most confident')


if __name__ == '__main__':
    main()







