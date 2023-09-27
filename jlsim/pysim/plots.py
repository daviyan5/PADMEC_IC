import numpy as np
import matplotlib.pyplot as plt

def plot_accurracy(cases):
    n = int(np.ceil(len(cases) / 2))
    m = int(np.ceil(len(cases) / n))
    fig, ax = plt.subplots(n, m, figsize = (11, 6))
    fig.suptitle("Comparação com Soluções Analíticas - Python")
    c = ["black", "blue", "green", "red", "grey"]
    for i, case in enumerate(cases):
        a, b = i // m, i % m
        ax[a, b].set_xscale("log")
        ax[a, b].set_yscale("log")
        name  = case["name"]
        error = case["error"]
        nvols = case["nvols"]
        q = -3 * (np.log(error[1:] / error[:-1]) / np.log(nvols[1:] / nvols[:-1]))
        ax[a, b].plot(nvols, error, color = c[i % len(c)], marker = "p", label = "I²rel")

        if name not in ["x + y + z", "x^2 + y^2 + z^2"]:
            minn = min(nvols)
            minerr = min(error)
            ax[a, b].plot([minn, 10*minn], [minerr, minerr * np.exp((-2/3) * np.log(10))], color = "r", label = "O(n²)")
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

def plot_times(cases):
    
    n = 2
    m = 2
    fig, ax = plt.subplots(2, 2, figsize = (11, 6))
    fig.suptitle("Comparação de Tempo de Execução - Python")
    c = ["black", "blue", "green", "red", "grey"]
    markers = ["p", "s", "o", "v", "D"]   
    new_keys = ["Total", "Pre-Processing", "System Preparation", "System Solving"]
    def prepare_times(case):
        times = case["times"]
        new_matrix = []
        for d in times:
            d["TPFA System Preparation"] += d["TPFA Boundary Conditions"]
            del d["TPFA Boundary Conditions"]
            del d["Post-Processing"]
            new_array = []
            for key in d:
                new_array.append(d[key])
            new_matrix.append(np.array(new_array))
        new_matrix = np.array(new_matrix)
        case["times"] = new_matrix

    
    for i, case in enumerate(cases):
        prepare_times(case)
        times = case["times"].T
        for j, vals in enumerate(times):
            nvols = case["nvols"]
            a, b = j // m, j % m
            ax[a, b].title.set_text(new_keys[j])
            ax[a, b].plot(nvols, vals, label = case["name"], color = c[i % len(c)], marker = markers[j % len(markers)])

    for i in range(2):
        for j in range(2):
            ax[i][j].grid()
            ax[i][j].legend(loc = "upper left")
            ax[i][j].set_xlabel("Número de Volumes")
            ax[i][j].set_ylabel("Tempo (s)")

    plt.tight_layout()
    plt.savefig("./tests/time_tests/times.png")
    plt.clf()

def plot_memory(cases):
    fig, ax = plt.subplots(1, 1, figsize = (11, 6))
    fig.suptitle("Comparação de Memória Utilizada - MATLAB")
    c = ["black", "blue", "green", "red", "grey"]
    markers = ["p", "s", "o", "v", "D"]
    for i, case in enumerate(cases):
        nvols = case["nvols"]
        memory = case["memory"]
        ax.plot(nvols, memory, label = case["name"], color = c[i % len(c)], marker = markers[i % len(markers)])
        
    ax.grid()
    ax.legend(loc = "upper left")
    ax.set_xlabel("Número de Volumes")
    ax.set_ylabel("Memória (MB)")

    plt.tight_layout()
    plt.savefig("./tests/memory_tests/memory.png")
    plt.clf()