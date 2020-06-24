import sys
import subprocess
import random
import numpy as np
import GPy
from nptyping import NDArray
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def forrester(x: NDArray[float]) -> NDArray[float]:
    # 最大化したいので符号を反転
    return (-1) * (6 * x - 2) ** 2 * np.sin(12 * x - 4)

def expected_improvement(X: NDArray[float], pred_mean: NDArray[float], pred_var: NDArray[float], xi: float = 0.0) -> NDArray[float]:
    return np.zeros(X.shape)

def experiment_each_seed(seed: int, initial_num: int, max_iter: int):
    '''
    初期点を生成する際のシードを引数のseedに固定したもとでベイズ最適化の実験を行う. 
    便利そうなので初期点の数はコマンドライン引数でしていできるようにしてみる. 

    :param seed: This is the seed value for generating initial data. 
    :param initial_num: This is the number of initial data. 
    '''

    # 定義域は[0, 1] でgrid_num分割して候補点を生成
    grid_num = 200 
    index_list = range(grid_num)
    X = np.c_[np.linspace(0, 1, grid_num)]
    y = forrester(X)

    random.seed(seed)
    # 初期点の生成
    train_index = random.sample(index_list, initial_num)
    X_train = X[train_index]
    y_train = y[train_index]

    # GP model
    kernel = GPy.kern.RBF(input_dim=X.shape[1], variance=1)
    model = GPy.models.GPRegression(X_train, y_train, kernel=kernel, normalizer=True)
    #観測誤差の分散は適当に固定
    noise_var = 1.0e-4
    model['.*Gaussian_noise.variance'].constrain_fixed(noise_var)
    model.optimize_restarts()
    print(model)
    pred_mean, pred_var = model.predict(X)

    # プロットや結果保存のためのディレクトリをつくる (実験結果をいつでも復元できるようにいろんなログはとっておいて損はない)
    result_dir_path = "./result/"
    _ = subprocess.check_call(["mkdir", "-p", result_dir_path + "seed_" + str(seed)])
    
    #回帰の様子をプロットしてみる (これは関数に切り出したほうがよさそうだが面倒なので今回はベタ書き)
    plt.plot(X.ravel(), y, "g--", label="true")
    plt.plot(X.ravel(), pred_mean, "b", label="pred mean")
    plt.fill_between(X.ravel(), (pred_mean + 2 * np.sqrt(pred_var)).ravel(), (pred_mean - 2 * np.sqrt(pred_var)).ravel(), alpha=0.3, color="blue")
    plt.plot(X_train.ravel(), y_train, "ro", label="observation")
    plt.legend(loc="lower left")
    plt.savefig(result_dir_path + "seed_"+str(seed)+"/predict_initial.pdf")
    plt.close()

    # simple regretを計算してlistで各イテレーションの推移を記録
    true_max = y.max()
    simple_regret = true_max - y_train.max()
    simple_regret_list = [simple_regret]

    # ベイズ最適化のイテレーションを回す
    for i in range(max_iter):
        acquisition_function = expected_improvement(X, pred_mean, pred_var)
        next_index = np.argmax(acquisition_function)
        x_next = X[next_index]
        y_next = y[next_index]
        X_train = np.append(X_train, [x_next], axis=0)
        y_train = np.append(y_train, [y_next], axis=0)
        #simple regret を計算
        simple_regret = true_max - y_train.max()
        simple_regret_list.append(simple_regret)
        #観測データを更新
        model.set_XY(X_train, y_train)
        model.optimize_restarts()
        pred_mean, pred_var = model.predict(X)

        plt.plot(X.ravel(), y, "g--", label="true")
        plt.plot(X.ravel(), pred_mean, "b", label="pred mean")
        plt.fill_between(X.ravel(), (pred_mean + 2 * np.sqrt(pred_var)).ravel(), (pred_mean - 2 * np.sqrt(pred_var)).ravel(), alpha=0.3, color="blue")
        plt.plot(X_train.ravel(), y_train, "ro", label="observation")
        plt.legend(loc="lower left")
        plt.savefig(result_dir_path + "seed_"+str(seed)+"/predict_" + str(i) +".pdf")
        plt.close()
    np.savetxt(result_dir_path + "seed_" + str(seed) + "/simple_regret.csv", np.array(simple_regret_list), delimiter=",")


def main():
    argv = sys.argv
    seed = 0
    initial_num = int(argv[1])
    max_iter = int(argv[2])
    # 初期点を変えた10通りの実験を並列に行う (詳しくは公式のリファレンスを見てください)
    parallel_num = 10
    _ = Parallel(n_jobs=parallel_num)([
        delayed(experiment_each_seed)(i, initial_num, max_iter) for i in range(parallel_num)
    ])

if __name__=="__main__":
    main()