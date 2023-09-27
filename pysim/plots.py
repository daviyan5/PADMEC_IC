import numpy as np
import matplotlib.pyplot as plt
import csv
from slugify import slugify

def plot_accurracy(cases):
    n = int(np.ceil(len(cases) / 2))
    m = int(np.ceil(len(cases) / n))
    fig, ax = plt.subplots(n, m, figsize = (11, 6))
    fig.suptitle("Comparação com Soluções Analíticas - Python")
    c = ["black", "blue", "green", "red", "grey"]
    qfile = csv.writer(open("./tests/q.csv", "w"))
    for i, case in enumerate(cases):
        a, b = i // m, i % m
        ax[a, b].set_xscale("log")
        ax[a, b].set_yscale("log")
        name  = case["name"]
        error = case["error"]
        nvols = case["nvols"]
        if case["name"] == "1/4 de Five-Spot":
            print("Error: ")
            for j in error:
                print("{:0.3e}".format(j), end=" ")
            print()
        if i == 0:
            first_row = ["Nome / Número de Volumes"]
            first_row.extend(nvols[1:])
            qfile.writerow(first_row)
        q = -3 * (np.log(error[1:] / error[:-1]) / np.log(nvols[1:] / nvols[:-1]))
        v = ["\"" + name + "\""]
        v.extend(np.round(q, decimals=4))
        if(name != "1/4 de Five-Spot"):
           qfile.writerow(v)
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
    fig, ax = plt.subplots(2, 2, figsize = (11, 8))
    fig.suptitle("Comparação de Tempo de Execução - Python")
    c = ["black", "blue", "green", "red", "grey"]
    markers = ["p", "s", "o", "v", "D"]   
    def prepare_times(case):
        times = case["times"]
        new_matrix = []
        prev_keys = []
        for d in times:
            d["TPFA System Preparation"] += d["TPFA Boundary Conditions"]
            d["Total Time"] -= d["Pre-Processing"]
            del d["TPFA Boundary Conditions"]
            del d["Post-Processing"]
            del d["Pre-Processing"]
            new_array = []
            for key in d:
                prev_keys.append(key)
                new_array.append(d[key])
            new_matrix.append(np.array(new_array))
        new_matrix = np.array(new_matrix)
        case["times"] = new_matrix
        return prev_keys

    gs = ax[1, 0].get_gridspec()
    ax[1, 1].remove()
    ax[1, 0].remove()
    axbig = fig.add_subplot(gs[1:, :])
    for i, case in enumerate(cases):
        prev_keys = prepare_times(case)
        new_keys = ["TPFA System Preparation", "TPFA System Solving", "Total Time"]
        new_keys_idx = []
        for key in new_keys:
            new_keys_idx.append(prev_keys.index(key))
        times = case["times"].T
        if case["name"] == "1/4 de Five-Spot":
            print("Times: ")
            print(list(case["nvols"]))
            for j in times[new_keys_idx[2]]:
                print("{:0.3e}".format(j), end=" ")
            print()
        for j, key in enumerate(new_keys):
            nvols = case["nvols"]
            k = new_keys_idx[j]
            a, b = j // m, j % m
            
            if key == "Total Time":
                axbig.set_facecolor("lightgrey")
                axbig.title.set_text(key)
                axbig.plot(nvols, times[k], label = case["name"], color = c[i % len(c)], marker = markers[j % len(markers)])
                axbig.grid(True)
                axbig.legend(loc = "upper left")
                axbig.set_xlabel("Número de Volumes")
                axbig.set_ylabel("Tempo (s)")

            else:
                ax[a, b].title.set_text(key)
                ax[a, b].plot(nvols, times[k], label = case["name"], color = c[i % len(c)], marker = markers[j % len(markers)])
                ax[a, b].grid(True)
                ax[a, b].legend(loc = "upper left")
                ax[a, b].set_xlabel("Número de Volumes")
                ax[a, b].set_ylabel("Tempo (s)")
                

    fig.tight_layout()
    plt.savefig("./tests/time_tests/times.png")
    plt.clf()

def plot_memory(cases):
    fig, ax = plt.subplots(1, 1, figsize = (11, 6))
    fig.suptitle("Comparação de Memória Utilizada - Python")
    c = ["black", "blue", "green", "red", "grey"]
    markers = ["p", "s", "o", "v", "D"]
    for i, case in enumerate(cases):
        nvols = case["nvols"]
        memory = case["memory"]
        if case["name"] == "1/4 de Five-Spot":
            print("Memory: ")
            for j in memory:
                print("{:0.3e}".format(j / 1e6), end=" ")
            print()
        ax.plot(nvols, memory / 1e6, label = case["name"], color = c[i % len(c)], marker = markers[i % len(markers)])
        
    ax.grid()
    ax.legend(loc = "upper left")
    ax.set_xlabel("Número de Volumes")
    ax.set_ylabel("Memória (MB)")

    plt.tight_layout()
    plt.savefig("./tests/memory_tests/memory.png")
    plt.clf()