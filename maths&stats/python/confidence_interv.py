import random as rand
import numpy as np
import matplotlib.pyplot as plt
rangel = int(input("size of max value (greater than 100): "))
if rangel <= 100:
    raise ValueError("Bad range")
elif type(rangel) != int:
    raise ValueError("Bad input type")
def data_randomiser(n):
    # Generates n random numbers between 0 and rangel
    if rangel < 1:
        raise ValueError("Bad range")

    return [rand.randint(0, rangel) for _ in range(n)]

def bootstrap(data, n):
    if not data:
        raise ValueError("Data cannot be empty")

    return np.random.choice(data, size=n, replace=True)

if __name__ == "__main__":
    data = data_randomiser(10)
    print(data)

    simulation_size = 100_000
    bootstrapped_data = bootstrap(data, simulation_size)
    tests = int(input("number of tests: "))
    actual_mean = bootstrapped_data.mean()
    print(f"Actual mean: {actual_mean:<12.4f}")
    loop=0
    sample_means = []  # ai changed layout to avoid OOM-kill since this file was made with limited processing power and population size is large
    #Data in the following while loop is very randomised
    while loop < tests:
        t_f = rand.randint(0,1)
        x = rand.randint(30,len(bootstrapped_data)-30)
        if t_f == 0:
            bootstrapped_data[x:x+45]
            sample_mean = (sum(bootstrapped_data[x:x+45]) / 45)
            print(f"Sample mean: {sample_mean:<12.4f}") 
        elif t_f == 1:
            bootstrapped_data[x-45:x]
            sample_mean = (sum(bootstrapped_data[x-45:x]) / 45)
            print(f"Sample mean: {sample_mean:<12.4f}")
        sample_means.append(sample_mean)  # NEW
        loop+=1

    # We use the sample_mean to plot points, a histogram showing the
    # bootstrapped population across [0, rangel], with every sample_mean
    # drawn as a thin vertical line on top of it.
    plt.figure(figsize=(10, 6))
    plt.hist(bootstrapped_data, bins=np.arange(0, rangel + 2) - 0.5,
             color="steelblue", edgecolor="none")
    for sm in sample_means:
        plt.axvline(sm, color="crimson", linewidth=0.5)
    plt.axvline(actual_mean, color="black", linewidth=1.5,
                label=f"actual mean = {actual_mean:.2f}")
    plt.xlim(0, rangel)
    plt.xlabel("value")
    plt.ylabel("frequency")
    plt.legend()
    plt.tight_layout()
    plt.draw()
    plt.savefig("my_plot.png")
