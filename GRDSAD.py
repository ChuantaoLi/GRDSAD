'''Graph Regularized Dynamic sampling AdaBoost'''
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y
from sklearn.exceptions import ConvergenceWarning
import warnings
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "8"
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def load_keel_dat(file_path):
    # This function loads a KEEL dataset from a .dat file
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if not line.startswith('@')]
    data = [line.split(',') for line in lines if line]
    df = pd.DataFrame(data)
    X = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').values
    y = df.iloc[:, -1].astype('category').cat.codes.values
    if len(np.unique(y)) != 2:
        raise ValueError("NOT BINARY CLASSIFICATION")
    return X, y


def calculate_gmean(y_true, y_pred):
    # Calculate the geometric mean of sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return np.sqrt(sens * spec)


class OverSample:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def _determine_clusters(self, X):
        # Use BIC to determine the number of clusters
        min_clusters = 2
        max_clusters = min(10, len(X) - 1)
        best_bic = np.inf
        best_n = 2
        for n in range(min_clusters, max_clusters + 1):
            gmm = GaussianMixture(n_components=n, random_state=self.random_state)
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_n = n
        return best_n

    def _generate_samples(self, X_min, clusters, n_samples):
        # Generate synthetic samples using linear interpolation
        synthetic = []
        rng = np.random.default_rng(self.random_state)
        for cluster_id in np.unique(clusters):
            cluster_data = X_min[clusters == cluster_id]
            if len(cluster_data) < 2:
                continue
            cluster_ratio = len(cluster_data) / len(X_min)
            n_cluster_samples = int(n_samples * cluster_ratio)
            for _ in range(n_cluster_samples):
                i1, i2 = rng.choice(len(cluster_data), 2, replace=False)
                alpha = rng.uniform()
                new_point = cluster_data[i1] + alpha * (cluster_data[i2] - cluster_data[i1])
                synthetic.append(new_point)
        if len(synthetic) == 0:
            return np.empty((0, X_min.shape[1]))
        return np.array(synthetic)[:n_samples]

    def fit_resample(self, X, y):
        X, y = check_X_y(X, y)
        classes = np.unique(y)
        majority_class = max(classes, key=lambda c: np.sum(y == c))
        minority_class = min(classes, key=lambda c: np.sum(y == c))
        X_min, X_maj = X[y == minority_class], X[y == majority_class]
        n_to_generate = len(X_maj) - len(X_min)
        if n_to_generate <= 0:
            return X, y
        n_clusters = self._determine_clusters(X_min)
        gmm = GaussianMixture(n_components=n_clusters, random_state=self.random_state)
        clusters = gmm.fit_predict(X_min)
        synthetic_X = self._generate_samples(X_min, clusters, n_to_generate)
        if len(synthetic_X) == 0:
            return X, y
        synthetic_y = np.full(len(synthetic_X), minority_class)
        return np.vstack([X, synthetic_X]), np.concatenate([y, synthetic_y])


def construct_graph(X, y, para):
    # A graph and its Laplacian matrix are constructed based on the input feature data and labels
    n = len(X)
    G = np.zeros((n, n))

    # Iterate over pairs of samples and compute the similarity between samples with the same label
    for i in range(n):
        for j in range(n):
            if y[i] == y[j]:
                dist = np.linalg.norm(X[i] - X[j]) ** 2
                G[i, j] = np.exp(-dist / para['rho'])  # Gaussian kernel to calculate similarity
    D = np.diag(G.sum(axis=1))
    L = D - G  # Laplacian matrix
    return G, L


def graph_regularized_projection(X, L, para):
    # The graph Laplacian matrix is used for feature dimensionality
    # Reduction to emphasize the local geometric structure of the data
    XLX = X.T @ L @ X
    A = XLX + para['gamma'] * np.eye(X.shape[1])
    eigvals, eigvecs = np.linalg.eigh(A)

    # Sort eigenvalues in ascending order (smallest first)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # If d is not specified, select dimensions capturing 95% of "energy"
    energy = np.cumsum(eigvals) / np.sum(eigvals)
    d = np.argmax(energy >= para['d']) + 1
    d = max(1, min(d, X.shape[1]))  # Ensure 1 <= d <= original dim

    # Select the smallest d eigenvalues (minimizing the Laplacian objective)
    return eigvecs[:, :d]


def stump_classify(data_matrix, dim, thresh, inequal):
    # Classify data based on a single feature and threshold
    ret = np.ones(data_matrix.shape[0])
    if inequal == 'lt':
        ret[data_matrix[:, dim] <= thresh] = -1.0
    else:
        ret[data_matrix[:, dim] > thresh] = -1.0
    return ret


def build_stump(X, y, D):
    # Build a decision stump
    m, n = X.shape
    best_stump = {}
    best_pred = np.zeros(m)
    min_err = np.inf

    for dim in range(n):
        for thresh in np.unique(X[:, dim]):
            for inequal in ('lt', 'gt'):
                pred = stump_classify(X, dim, thresh, inequal)
                err = (pred != y).astype(float)
                weighted_err = np.dot(D, err)
                if weighted_err < min_err:
                    min_err = weighted_err
                    best_stump = {'dim': dim, 'thresh': thresh, 'ineq': inequal}
                    best_pred = pred.copy()
    return best_stump, min_err, best_pred


def compute_group_weights(H, iteration, num_iter, para):
    # Compute group weights based on the current iteration and the number of iterations
    boundary_threshold = 0.6
    bins = [0, 0.4, boundary_threshold, 1.0]
    groups = np.digitize(H, bins) - 1
    groups = np.clip(groups, 0, 2)

    C_bar = np.zeros(3)
    for j in range(3):
        mask = (groups == j)
        if np.any(mask):
            C_bar[j] = H[mask].mean()

    i, n = iteration, num_iter
    alpha = np.tan((i - 1) * np.pi / (2 * n))
    beta = 1 / (1 + np.exp(-para['eta'] * ((2 * i / n) - 1)))

    delta = np.zeros(3)
    for j in range(3):
        if (C_bar[j] < 0.4) or (C_bar[j] > boundary_threshold):
            delta[j] = alpha
        else:
            delta[j] = beta * alpha

    w = 1.0 / (C_bar + delta + 1e-8)
    w_norm = w / w.sum()
    return groups, w_norm


def ada_boost_train_dynamic(X, y, num_iter, random_state, para):
    # Train the AdaBoost classifier with dynamic sampling
    classes = np.unique(y)
    pos, neg = classes[0], classes[1]
    y_enc = np.where(y == pos, 1, -1)
    label_map = {1: pos, -1: neg}
    m = X.shape[0]
    agg_est = np.zeros(m)
    classifiers, betas = [], []

    # Calculate the class imbalance ratio
    classes, counts = np.unique(y, return_counts=True)
    minority_class = classes[np.argmin(counts)]
    majority_class = classes[np.argmax(counts)]
    N_minority = sum(y == minority_class)
    N_majority = sum(y == majority_class)
    imbalance_ratio = N_majority / N_minority

    for i in range(1, num_iter + 1):
        p = 1 / (1 + np.exp(-agg_est))
        P_correct = np.where(y_enc == 1, p, 1 - p)
        P_wrong = 1 - P_correct
        H = 1 - (P_correct - P_wrong)
        H = H / 2.0

        groups, w_norm = compute_group_weights(H, i, num_iter, para)

        if imbalance_ratio > 4:
            Ntarget = int(N_minority * 1.2)
        else:
            Ntarget = N_minority

        maj_idx = np.where(y_enc == -1)[0]
        min_idx = np.where(y_enc == 1)[0]
        maj_groups = groups[maj_idx]
        sampled_maj = []

        # Sample each group based on its sample confidence
        for j in range(3):
            group_idx = maj_idx[maj_groups == j]
            Nj = int(Ntarget * w_norm[j])
            if Nj > 0 and len(group_idx) > 0:
                replace = len(group_idx) < Nj
                sampled = np.random.choice(group_idx, Nj, replace=replace)
                sampled_maj.extend(sampled)

        # If the number of samples is insufficient, replenish the remaining samples
        if len(sampled_maj) < Ntarget:
            remaining = np.setdiff1d(maj_idx, sampled_maj)
            sampled_maj.extend(np.random.choice(remaining, Ntarget - len(sampled_maj), replace=True))

        X_group = np.vstack([X[sampled_maj], X[min_idx]])
        y_group = np.concatenate([y_enc[sampled_maj], y_enc[min_idx]])

        overSampler = OverSample(random_state=random_state)
        try:
            X_res, y_res = overSampler.fit_resample(X_group, y_group)
        except ValueError:
            X_res, y_res = X_group, y_group

        # Build graph and Laplacian matrix, perform graph regularized projection
        G, L = construct_graph(X_res, y_res, para)
        P = graph_regularized_projection(X_res, L, para)
        X_proj = X_res @ P
        X_full_proj = X @ P

        D_sub = np.ones(len(y_res)) / len(y_res)
        stump, error, _ = build_stump(X_proj, y_res, D_sub)
        error = max(error, 1e-16)
        beta = 0.5 * np.log((1 - error) / error)

        classifiers.append({**stump, 'beta': beta, 'P': P})
        betas.append(beta)

        pred = stump_classify(X_full_proj, stump['dim'], stump['thresh'], stump['ineq'])
        agg_est += beta * pred  # Weighted combination of weak classifiers

    return classifiers, betas, label_map


def ada_classify(X, classifiers, betas, label_map):
    # Classify data using the trained classifiers
    agg = np.zeros(X.shape[0])
    for clf in classifiers:
        P = clf['P']
        X_proj = X @ P
        pred = stump_classify(X_proj, clf['dim'], clf['thresh'], clf['ineq'])
        agg += clf['beta'] * pred
    pred_enc = np.sign(agg)
    return np.where(pred_enc == 1, label_map[1], label_map[-1])


def run_repeated_holdout(X, y, random_state, repeat, test_size, para):
    # Run repeated holdout cross-validation for Experiment1 datasets experiment
    metrics = {'gmean': [], 'auc': []}

    for i in range(repeat):
        rs = random_state + i
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rs, stratify=y)
        classifiers, betas, label_map = ada_boost_train_dynamic(X_train, y_train, 30, rs, para)
        preds = ada_classify(X_test, classifiers, betas, label_map)

        metrics['gmean'].append(calculate_gmean(y_test, preds))
        metrics['auc'].append(roc_auc_score(y_test, preds))

        print(f"[Run {i + 1}] GMean={metrics['gmean'][-1]:.4f}, AUC={metrics['auc'][-1]:.4f}")

    print("\n=== Final Averaged Results ===")
    results = {}
    for k in metrics:
        mean, std = np.mean(metrics[k]), np.std(metrics[k])
        results[k] = (mean, std)
        print(f"{k.title()}: {mean:.3f}±{std:.3f}")
    return results


def run_repeated_holdout_have_train_test(X_train, y_train, X_test, y_test, para):
    # For KEEL datasets, because the training and testing sets are already given
    print(para)
    metrics = {'gmean': [], 'auc': []}

    clf, betas, lm = ada_boost_train_dynamic(X_train, y_train, 30, 42, para)
    preds = ada_classify(X_test, clf, betas, lm)

    metrics['gmean'].append(calculate_gmean(y_test, preds))
    metrics['auc'].append(roc_auc_score(y_test, preds))

    print(f"GMean={metrics['gmean'][-1]:.4f}, AUC={metrics['auc'][-1]:.4f}")

    return {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}


if __name__ == '__main__':
    '''Read Experiment1 datasets'''
    # data = pd.read_csv('Experiment1/10Ydata.csv')
    # data = data.dropna()
    # X = data.iloc[:, :-1].values
    # y = data.iloc[:, -1].values
    # para = {'eta': 0.1, 'rho': 1, 'gamma': 1.5, 'd': 0.3}
    # results = run_repeated_holdout(X, y, 42, 5, 0.3, para)

    '''Read Experiment2 datasets'''
    datasets = [
        {'name': 'ecoli-0_vs_1', 'params': {'eta': 0.1, 'rho': 1, 'gamma': 1, 'd': 0.1}},
        {'name': 'ecoli1', 'params': {'eta': 0.1, 'rho': 1, 'gamma': 1, 'd': 0.8}},
        {'name': 'ecoli2', 'params': {'eta': 0.5, 'rho': 1.2, 'gamma': 0.1, 'd': 0.5}},
        {'name': 'ecoli3', 'params': {'eta': 0.1, 'rho': 1, 'gamma': 0.1, 'd': 0.8}},
        {'name': 'haberman', 'params': {'eta': 0.3, 'rho': 1.4, 'gamma': 0.1, 'd': 0.4}},
        {'name': 'new-thyroid1', 'params': {'eta': 0.4, 'rho': 1.4, 'gamma': 0.05, 'd': 0.1}},
        {'name': 'new-thyroid2', 'params': {'eta': 0.1, 'rho': 1, 'gamma': 0.05, 'd': 0.8}},
        {'name': 'yeast3', 'params': {'eta': 0.5, 'rho': 2, 'gamma': 0.5, 'd': 0.3}}
    ]

    all_results = {}


    def format_result(values):
        return f"{np.mean(values):.3f}±{np.std(values):.3f}"


    def run_repeated_holdout_have_train_test(X_train, y_train, X_test, y_test, para):
        classifiers, betas, label_map = ada_boost_train_dynamic(
            X_train, y_train,
            num_iter=30,
            random_state=42,
            para=para
        )

        preds = ada_classify(X_test, classifiers, betas, label_map)
        return {
            'gmean': [calculate_gmean(y_test, preds)],
            'auc': [roc_auc_score(y_test, preds)]
        }


    for dataset in datasets:
        print(f'\n==== Processing Dataset: {dataset["name"]} ====')
        print(f'Parameters: {dataset["params"]}')

        data_folder = f'Experiment2/{dataset["name"]}-5-fold'
        ggmean, aauc = [], []

        for i in range(1, 6):
            X_train, y_train = load_keel_dat(os.path.join(data_folder, f'{dataset["name"]}-5-{i}tra.dat'))
            X_test, y_test = load_keel_dat(os.path.join(data_folder, f'{dataset["name"]}-5-{i}tst.dat'))

            results = run_repeated_holdout_have_train_test(X_train, y_train, X_test, y_test, dataset['params'])
            ggmean.append(results['gmean'][0])
            aauc.append(results['auc'][0])
            print(f'Fold {i}: G-Mean={results["gmean"][0]:.3f}, AUC={results["auc"][0]:.3f}')

        all_results[dataset["name"]] = {
            'G-Mean': ggmean,
            'AUC': aauc,
            'Parameters': dataset['params']
        }

    print('\n' + '=' * 50)
    print('Final Results (Mean±Std, 3 decimal places)')
    print('=' * 50)

    results_table = []
    for name, res in all_results.items():
        results_table.append({
            'Dataset': name,
            'Parameters': str(res['Parameters']),
            'G-Mean': format_result(res['G-Mean']),
            'AUC': format_result(res['AUC'])
        })

    df = pd.DataFrame(results_table)
    print(df[['Dataset', 'G-Mean', 'AUC']].to_string(index=False))
