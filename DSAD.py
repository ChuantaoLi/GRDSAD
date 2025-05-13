'''Dynamic oversampling and dynamic undersampling'''
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y
from sklearn.model_selection import train_test_split
import warnings
from sklearn.exceptions import ConvergenceWarning
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


class OverSample(BaseEstimator, TransformerMixin):
    # Initialize method
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.valid_clusters_ = []

    # The optimal number of clusters is automatically determined using BIC
    def _determine_clusters(self, X):
        min_clusters = 2
        max_clusters = min(10, len(X) - 1)
        best_bic = np.inf

        for n in range(min_clusters, max_clusters + 1):
            gmm = GaussianMixture(n_components=n, random_state=self.random_state)
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_n = n
        return best_n

    # New minority samples are synthesized according to the clustering results
    def _generate_samples(self, X_min, clusters, n_samples):
        synthetic = []
        rng = check_random_state(self.random_state)

        for cluster_id in np.unique(clusters):
            cluster_data = X_min[clusters == cluster_id]
            if len(cluster_data) < 2:
                continue

            # Calculate (proportionally) the number of samples that should be synthesized for this cluster
            cluster_ratio = len(cluster_data) / len(X_min)
            n_cluster_samples = int(n_samples * cluster_ratio)

            # Randomly select two samples from the cluster for linear interpolation
            for _ in range(n_cluster_samples):
                i1, i2 = rng.choice(len(cluster_data), 2, replace=False)
                point1 = cluster_data[i1]
                point2 = cluster_data[i2]
                alpha = rng.uniform(0, 1)
                new_point = point1 + alpha * (point2 - point1)
                synthetic.append(new_point)

        return np.array(synthetic)[:n_samples]

    def fit_resample(self, X, y):
        X, y = check_X_y(X, y)
        classes = np.unique(y)

        majority_class = max(classes, key=lambda c: np.sum(y == c))
        minority_class = min(classes, key=lambda c: np.sum(y == c))

        X_min = X[y == minority_class]
        X_maj = X[y == majority_class]

        n_min = len(X_min)
        n_maj = len(X_maj)
        n_to_generate = n_maj - n_min

        n_clusters = self._determine_clusters(X_min)
        gmm = GaussianMixture(n_components=n_clusters, random_state=self.random_state)
        clusters = gmm.fit_predict(X_min)

        synthetic_X = self._generate_samples(X_min, clusters, n_to_generate)
        synthetic_y = np.full(len(synthetic_X), minority_class)

        X_res = np.vstack([X, synthetic_X])
        y_res = np.concatenate([y, synthetic_y])

        return X_res, y_res


def stump_classify(data_matrix, dim, thresh, inequal):
    # Classify data based on a single feature and threshold
    m = data_matrix.shape[0]
    ret = np.ones(m)
    if inequal == 'lt':
        ret[data_matrix[:, dim] <= thresh] = -1.0
    else:
        ret[data_matrix[:, dim] > thresh] = -1.0
    return ret


def build_stump(X, y, D):
    # Build a decision stump
    m, n = X.shape
    best_stump = {}
    best_est = np.zeros(m)
    min_err = np.inf

    for dim in range(n):
        vals = np.unique(X[:, dim])
        for thresh in vals:
            for inequal in ('lt', 'gt'):
                pred = stump_classify(X, dim, thresh, inequal)
                err = (pred != y).astype(float)
                weighted_err = np.dot(D, err)
                if weighted_err < min_err:
                    min_err = weighted_err
                    best_est = pred.copy()
                    best_stump = {'dim': dim, 'thresh': thresh, 'ineq': inequal}
    return best_stump, min_err, best_est


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


def ada_boost_train_dynamic(X, y, num_iter, para):
    # Train the AdaBoost classifier with dynamic sampling
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError('NOT BINARY CLASSIFICATION')

    pos, neg = classes[0], classes[1]
    y_enc = np.where(y == pos, 1, -1)
    label_map = {1: pos, -1: neg}
    m = X.shape[0]
    D = np.ones(m) / m
    agg_est = np.zeros(m)
    classifiers, betas = [], []

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
        N_minority = sum(y == minority_class)
        if imbalance_ratio > 4:
            Ntarget = int(N_minority * 1.2)
        else:
            Ntarget = N_minority

        maj_idx = np.where(y_enc == -1)[0]
        maj_groups = groups[maj_idx]
        sampled_maj = []

        for j in range(3):
            group_idx = maj_idx[maj_groups == j]
            Nj = int(Ntarget * w_norm[j])
            if Nj > 0 and len(group_idx) > 0:
                sampled = np.random.choice(group_idx, min(Nj, len(group_idx)), replace=False)
                sampled_maj.extend(sampled)

        # If not enough, fill in the sample
        if len(sampled_maj) < Ntarget:
            remaining = np.setdiff1d(maj_idx, sampled_maj)
            sampled_maj.extend(np.random.choice(remaining, Ntarget - len(sampled_maj), replace=False))

        min_samples = np.where(y_enc == 1)[0]
        X_min, y_min = X[min_samples], y_enc[min_samples]
        overSample = OverSample(random_state=42)
        try:
            X_res, y_res = overSample.fit_resample(np.vstack([X[sampled_maj], X_min]), np.concatenate([y_enc[sampled_maj], y_min]))
        except ValueError:
            X_res, y_res = X, y

        D_sub = np.ones(len(y_res)) / len(y_res)
        stump, error, _ = build_stump(X_res, y_res, D_sub)
        error = max(error, 1e-16)
        beta = 0.5 * np.log((1 - error) / error)
        classifiers.append({**stump, 'beta': beta})
        betas.append(beta)

        class_est_full = stump_classify(X, stump['dim'], stump['thresh'], stump['ineq'])
        agg_est += beta * class_est_full
        D *= np.exp(-beta * y_enc * class_est_full)
        D /= D.sum()

    return classifiers, betas, label_map


def ada_classify(X, classifiers, betas, label_map):
    # Classify data using the trained classifiers
    agg = np.zeros(X.shape[0])
    for stump, beta in zip(classifiers, betas):
        agg += beta * stump_classify(X, stump['dim'], stump['thresh'], stump['ineq'])
    pred_enc = np.sign(agg)
    return np.where(pred_enc == 1, label_map[1], label_map[-1])


def calculate_gmean(y_true, y_pred):
    # Calculate the geometric mean of sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    return np.sqrt(sens * spec)


if __name__ == '__main__':

    data = pd.read_csv(r'Experiment1/Framingham.csv')
    data = data.dropna()
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    para = {'eta': 0.3, 'rho': 1.1, 'gamma': 3, 'd': 0.5}

    results = {
        'gmean': [],
        'auc': []
    }

    for run in range(5):
        print(f'====  {run + 1} Iteration ====')

        random_seed = run
        np.random.seed(random_seed)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed)

        classifiers, betas, label_map = ada_boost_train_dynamic(X_train, y_train, 30, para)
        preds = ada_classify(X_test, classifiers, betas, label_map)

        gmean = calculate_gmean(y_test, preds)
        auc = roc_auc_score(y_test, preds)

        results['gmean'].append(gmean)
        results['auc'].append(auc)

        print(f'G-Mean: {gmean:.4f}')
        print(f'AUC: {auc:.4f}')
        print()

    stats_data = []
    for metric, scores in results.items():
        stats_data.append({
            'Metric': metric.capitalize(),
            'Mean': np.mean(scores),
            'Std': np.std(scores)
        })

    stats_df = pd.DataFrame(stats_data)

    raw_data = []
    for run in range(5):
        for metric in results.keys():
            raw_data.append({
                'Run': run + 1,
                'Metric': metric.capitalize(),
                'Value': results[metric][run]
            })

    raw_df = pd.DataFrame(raw_data)

    print('==== Result ====')
    for _, row in stats_df.iterrows():
        metric = row['Metric']
        mean = f"{row['Mean']:.3f}"
        std = f"{row['Std']:.3f}"
        print(f"{metric}: {mean}±{std}")

# if __name__ == '__main__':
#     data_folder = 'Experiment2/ecoli-0_vs_1-5-fold'
#
#     # 初始化结果存储
#     final_results = {
#         'G-Mean': [],
#         'AUC': []
#     }
#
#     # 5折交叉验证
#     for i in range(1, 6):
#         print(f'==== Fold {i} ====')
#
#         # 加载数据
#         train_file = os.path.join(data_folder, f'ecoli-0_vs_1-5-{i}tra.dat')
#         test_file = os.path.join(data_folder, f'ecoli-0_vs_1-5-{i}tst.dat')
#         X_train, y_train = load_keel_dat(train_file)
#         X_test, y_test = load_keel_dat(test_file)
#
#         # 训练和预测
#         classifiers, betas, label_map = ada_boost_train_dynamic(
#             X_train, y_train,
#             num_iter=30
#         )
#         preds = ada_classify(X_test, classifiers, betas, label_map)
#
#         # 计算指标
#         gmean = calculate_gmean(y_test, preds)
#         auc = roc_auc_score(y_test, preds)
#
#         final_results['G-Mean'].append(gmean)
#         final_results['AUC'].append(auc)
#
#         print(f'G-Mean: {gmean:.3f}')
#         print(f'AUC: {auc:.3f}\n')
#
#     # 计算统计结果
#     stats_data = []
#     for metric, values in final_results.items():
#         mean_val = np.mean(values)
#         std_val = np.std(values)
#         stats_data.append({
#             'Metric': metric,
#             'Mean ± Std': f"{mean_val:.3f} ± {std_val:.3f}",
#             'Mean': f"{mean_val:.3f}",
#             'Std': f"{std_val:.3f}"
#         })
#
#     # 创建DataFrame
#     stats_df = pd.DataFrame(stats_data)
#     raw_df = pd.DataFrame([
#         {'Fold': i + 1, 'Metric': metric, 'Value': f"{final_results[metric][i]:.3f}"}
#         for i in range(5) for metric in final_results.keys()
#     ])
#
#     # 打印结果
#     print('==== Final Results ====')
#     print(stats_df[['Metric', 'Mean ± Std']].to_string(index=False))
#     print('\n==== Detailed Results ====')
#     print(raw_df.to_string(index=False))
