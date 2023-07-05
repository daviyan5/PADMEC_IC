import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy(res : dict, vols : np.ndarray, vols2Well : np.ndarray):
    vols = np.array(vols)
    vols2Well = np.array(vols2Well)
    n = int(np.ceil(len(res) / 2))
    m = int(np.ceil(len(res) / n))
    names = list(res.keys())
    fig, ax = plt.subplots(n, m, figsize = (11, 6))
    fig.suptitle("Comparação com Soluções Analíticas - JULIA")
    c = ["black", "blue", "green", "red", "grey"]

    for i, name in enumerate(names):
        a, b = i // m, i % m
        ax[a, b].set_xscale("log")
        ax[a, b].set_yscale("log")
        res[name] = np.mean(res[name], axis = 1)
        pvols = vols if name != "2Well" else vols2Well
        res[name] = res[name][:len(pvols)]
        ax[a, b].plot(pvols, res[name], color = c[i % len(c)], marker = "p", label = "I²rel")
        scale_vols = res[name][0] * (pvols[0] ** 2) / (pvols ** 2)
        if i <= 1:
            ax[a][b].plot(pvols, scale_vols, label = "O(n²)", color = 'purple', linestyle = "--") 
        else:
            ax[a, b].set_ylim(1e-20, 1e-10)
        ax[a, b].set_title(name)
        ax[a, b].set_xlabel("Número de Volumes")
        ax[a, b].set_ylabel("I²rel")
        ax[a, b].legend()
        ax[a, b].grid()
    plt.tight_layout()
    plt.savefig("./tests/accuracy_tests/accuracy.png")
    plt.clf()

def plot_times(res : dict, vols : np.array, vols2Well : np.ndarray):
    vols = np.array(vols)
    vols2Well = np.array(vols2Well)
    n = 2
    m = 2

    names = list(res.keys())
    fig, ax = plt.subplots(2, 2, figsize = (11, 6))
    fig.suptitle("Comparação de Tempo de Execução - JULIA")
    c = ["black", "blue", "green", "red", "grey"]
    markers = ["p", "s", "o", "v", "D"]    
    for i, name in enumerate(names):
        
        for j, key in enumerate(res[name]):
            pvols = vols if name != "2Well" else vols2Well
            res[name][key] = res[name][key][:len(pvols)]
            res[name][key] = np.mean(res[name][key], axis = 1)
            a, b = j // m, j % m
            ax[a, b].title.set_text(key)
            ax[a, b].plot(pvols, res[name][key], label = name, color = c[i % len(c)], marker = markers[j % len(markers)])
    for i in range(2):
        for j in range(2):
            ax[i][j].grid()
            ax[i][j].legend(loc = "upper left")
            ax[i][j].set_xlabel("Número de Volumes")
            ax[i][j].set_ylabel("Tempo (s)")
    plt.tight_layout()
    plt.savefig("./tests/time_tests/times.png")
    plt.clf()

def plot_memory(res : dict, vols : np.array, vols2Well : np.ndarray):
    vols = np.array(vols)
    vols2Well = np.array(vols2Well)
    fig, ax = plt.subplots(1, 1, figsize = (11, 6))
    fig.suptitle("Comparação de Memória Utilizada - JULIA")
    c = ["black", "blue", "green", "red", "grey"]
    markers = ["p", "s", "o", "v", "D"]
    for i, name in enumerate(res):
        pvols = vols if name != "2Well" else vols2Well
        res[name] = res[name][:len(pvols)]
        res[name] = np.mean(res[name], axis = 1)
        ax.plot(pvols, res[name], label = name, color = c[i % len(c)], marker = markers[i % len(markers)])
    ax.grid()
    ax.legend(loc = "upper left")
    ax.set_xlabel("Número de Volumes")
    ax.set_ylabel("Memória (MB)")

    plt.tight_layout()
    plt.savefig("./tests/memory_tests/memory.png")
    plt.clf()