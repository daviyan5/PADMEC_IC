import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat
import csv
from slugify import slugify


def plot_accuracy(res : dict, vols : np.ndarray, volsQFiveSpot : np.ndarray):
    vols = np.array(vols)
    volsQFiveSpot = np.array(volsQFiveSpot)
    n = int(np.ceil(len(res) / 2))
    m = int(np.ceil(len(res) / n))
    names = list(res.keys())
    fig, ax = plt.subplots(n, m, figsize = (11, 6))
    fig.suptitle("Comparação com Soluções Analíticas - MATLAB")
    c = ["black", "blue", "green", "red", "grey"]
    qfile = csv.writer(open("./tests/q.csv", "w"))
    for i, name in enumerate(names):
        a, b = i // m, i % m
        ax[a, b].set_xscale("log")
        ax[a, b].set_yscale("log")
        pvols = vols if name != "1/4 de Five-Spot" else volsQFiveSpot
        res[name] = res[name][:len(pvols)]
        if i == 0:
            first_row = ["Nome / Número de Volumes"]
            first_row.extend(pvols[1:])
            qfile.writerow(first_row)
        with np.errstate(divide='ignore'):
            q = -3 * (np.log(res[name][1:] / res[name][:-1]) / np.log(pvols[1:] / pvols[:-1]))
        q = q[~np.isnan(q)]
        v = ["\"" + name + "\""]
        v.extend(np.round(q, decimals=4))
        if(name != "1/4 de Five-Spot"):
           qfile.writerow(v)
        else:
            print(pvols)

        ax[a, b].plot(pvols, res[name], color = c[i % len(c)], marker = "p", label = "I²rel")
       
        if name != "x + y + z" and name != "x^2 + y^2 + z^2":
            minn = min(pvols)
            minerr = min(res[name])
            ax[a, b].plot([minn, 10*minn], [minerr, minerr * np.exp((-2/3) * np.log(10))], color = "r", label = "O")
        else:
            ax[a, b].set_ylim(1e-20, 1e-10)
        ax[a, b].set_title(name)
        ax[a, b].set_xlabel("Número de Volumes")
        ax[a, b].set_ylabel("I²rel")
        ax[a, b].legend()
        ax[a, b].grid()
    plt.tight_layout()
    plt.savefig("./tests/accuracy.png")
    plt.clf()

def plot_times(res : dict, vols : np.array, volsQFiveSpot : np.ndarray):
    vols = np.array(vols)
    volsQFiveSpot = np.array(volsQFiveSpot)
    n = 2
    m = 2

    names = list(res.keys())
    fig, ax = plt.subplots(2, 2, figsize = (11, 8))
    fig.suptitle("Comparação de Tempo de Execução - MATLAB")
    c = ["black", "blue", "green", "red", "grey"]
    markers = ["p", "s", "o", "v", "D"]    
    gs = ax[1, 0].get_gridspec()
    ax[1, 1].remove()
    ax[1, 0].remove()
    axbig = fig.add_subplot(gs[1:, :])
    
    for i, name in enumerate(names):
        new_keys = ["Pre-Processing", "System Preparation", "TPFA Solver", "Total"]
        del res[name]["Pre-Processing*"]
        
        for j, key in enumerate(res[name]):
            pvols = vols if name != "1/4 de Five-Spot" else volsQFiveSpot
            res[name][key] = res[name][key][:len(pvols)]
            a, b = j // m, j % m
            if name == "1/4 de Five-Spot" and key == "Total":
                print("Times: ")
                for k in res[name][key]:
                    print("{:0.3e}".format(k), end=" ")
                print()
            if key == "Total":
                axbig.set_facecolor("lightgrey")
                axbig.title.set_text("Total Time")
                axbig.plot(pvols, res[name][key], label = name, color = c[i % len(c)], marker = markers[j % len(markers)])
                axbig.grid(True)
                axbig.legend(loc = "upper left")
                axbig.set_xlabel("Número de Volumes")
                axbig.set_ylabel("Tempo (s)")

            else:
                ax[a, b].title.set_text(key)
                ax[a, b].plot(pvols, res[name][key], label = name, color = c[i % len(c)], marker = markers[j % len(markers)])
                ax[a, b].grid(True)
                ax[a, b].legend(loc = "upper left")
                ax[a, b].set_xlabel("Número de Volumes")
                ax[a, b].set_ylabel("Tempo (s)")
    
    plt.tight_layout()
    plt.savefig("./tests/times.png")
    plt.clf()

def plot_memory(res : dict, vols : np.array, volsQFiveSpot : np.ndarray):
    vols = np.array(vols)
    volsQFiveSpot = np.array(volsQFiveSpot)
    fig, ax = plt.subplots(1, 1, figsize = (11, 6))
    fig.suptitle("Comparação de Memória Utilizada - MATLAB")
    c = ["black", "blue", "green", "red", "grey"]
    markers = ["p", "s", "o", "v", "D"]
    for i, name in enumerate(res):
        pvols = vols if name != "1/4 de Five-Spot" else volsQFiveSpot
        res[name] = res[name][:len(pvols)]
        if name == "1/4 de Five-Spot":
            print("Memory: ")
            for j in res[name]:
                print("{:0.3e}".format(j / 1e6), end=" ")
            print()
        ax.plot(pvols, res[name] / 1e6, label = name, color = c[i % len(c)], marker = markers[i % len(markers)])
    ax.grid()
    ax.legend(loc = "upper left")
    ax.set_xlabel("Número de Volumes")
    ax.set_ylabel("Memória (MB)")

    plt.tight_layout()
    plt.savefig("./tests/memory.png")
    plt.clf()


def main():
    keys = ["Pre-Processing*", "System Preparation", "TPFA Solver", "Total"]
    case1 = loadmat("./vtks/linear_case.mat")
    case2 = loadmat("./vtks/extra_case1.mat")
    case3 = loadmat("./vtks/extra_case2.mat")
    case4 = loadmat("./vtks/qfive_spot.mat")
    case5 = loadmat("./vtks/quadratic_case.mat")
    case1["name"] = "x + y + z"
    case2["name"] = "sin(x) + cos(y) + exp(z)"
    case3["name"] = "(x + 1) * log(1 + x) + 1/(y + 1) + z^2"
    case4["name"] = "1/4 de Five-Spot"
    case5["name"] = "x^2 + y^2 + z^2"
    cases = [case1, case5, case3, case4]
    vols = np.squeeze(case1["nvols"])
    volsQFiveSpot = np.squeeze(case4["nvols"])
    def run_from_cases(cases):
        res_time = {}
        res_mem = {}
        res_acc = {}
        for case in cases:
            res_time[case["name"]] =  dict(zip(keys, case["time"].T))
            res_mem[case["name"]] = case["memory"]
            res_acc[case["name"]] = case["error"]
        for key in res_time:
            res_acc[key] = np.squeeze(res_acc[key])
            res_mem[key] = np.squeeze(res_mem[key])
        return res_time, res_mem, res_acc
        
    res_time, res_mem, res_acc = run_from_cases(cases)
    plot_accuracy(res_acc, vols, volsQFiveSpot)
    cases = [case1, case5, case3, case4]
    res_time, res_mem, res_acc = run_from_cases(cases)
    plot_times(res_time, vols, volsQFiveSpot)
    plot_memory(res_mem, vols, volsQFiveSpot)

main()