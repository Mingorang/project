import math
from math import factorial
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets

# Set up choices for distributions 
print("Binomial")
print("Normal ")
print("Poisson ")
print("Geometric ")
print("Negative binomial")
choice = int(input("Choose a dist: "))
pdfcdf = bool(input("CDF or PDF (y/n): ").strip().lower().startswith('y'))

patches = []
y = []
X_vals = [] 
a = 0

# Binomial - discrete
if choice == 1:
    plt.yscale('log')
    discrete_check = True
    n = int(input("Number of trials: "))
    p = float(input("Probability of success (0-1): "))
    x = int(input("How many successes happened: "))
    bino_cdf = 0
    k = 0

    if x > n or (p > 1 or p < 0):
        sys.exit("Invalid: successes cannot be greater than the number of trials and probability of successes must be in the range 0 to 1.")
        
    while k < (x + 1):
        bino_pdf = (factorial(n) / (factorial(k) * factorial(n - k))) * (p**k) * ((1 - p)**(n - k))
        bino_cdf += bino_pdf
        print(f"Successes (k): {k} | PDF: {bino_pdf:.20f} | Running CDF: {bino_cdf:.20f}")
        y.append(bino_pdf)
        X_vals.append(k)
        k += 1
        
    print(f"\nFinal Cumulative Distribution Function (CDF): {bino_cdf:.16f}")
    print(f"\nFinal Probability Density Function (PDF): {(factorial(n) / (factorial(x) * factorial(n - x))) * (p**x) * ((1 - p)**(n - x)):.16f}")

# Normal - continuous
elif choice == 2:
    discrete_check = False
    μ = float(input("Choose a value for mean: "))     
    σ = float(input("Choose a value for standard deviation: "))
    x = int(float(input("no. of successes / occurences: ")))
    cdf_norm = 0
    
    if pdfcdf == True:
        a = float(input("Lower/Upper limit: "))
        steps = 1000
        if x == a:
            sys.exit("You should have chosen PDF then.")
        else:
            step = (x - a) / steps
            
        while (a <= x and step > 0) or (a >= x and step < 0):
            pdf_norm = (1 / (σ * math.sqrt(2 * math.pi))) * (math.e ** (-1*((a-μ)**2)/(2*σ**2)))
            cdf_norm += pdf_norm * abs(step)
            print(f" Occurences (a): {a:>12.4g} | PDF: {pdf_norm:>12.6g} | Running CDF: {cdf_norm:>12.6g}")
            y.append(pdf_norm)
            X_vals.append(a)
            a += step
    else:
        pdf_norm = (1 / (σ * math.sqrt(2 * math.pi))) * (math.e ** (-1*((x-μ)**2)/(2*σ**2)))
        print(f"A value of {x} in this distribution has a PDF of {pdf_norm:>18.15f}")

# Poisson - discrete
elif choice == 3:
    plt.yscale('log')
    λ = int(input("Value for poisson paramter: "))
    x = int(input("no. of successes / occurences: "))
    pois_cdf = 0
    k = 0
    
    if pdfcdf == True:
        while k <= x:
            pois_pdf = ((math.e)**(-1*λ))*(λ**k)*(1/(factorial(k)))
            pois_cdf += pois_pdf
            print(f"Successes (k): {k:>4.2g} | PDF: {pois_pdf:>12.6g} | Running CDF: {pois_cdf:12.6g}")
            y.append(pois_pdf)
            X_vals.append(k)
            k += 1
        print(f"\nFinal Cumulative Distribution Function (CDF): {pois_cdf:16.15g}")
    else:
        pois_pdf = ((math.e)**(-1*λ))*(λ**x)*(1/(factorial(x)))
        print(f"X : {x} | PDF : {pois_pdf:10.9g}")

# Geometric - discrete
elif choice == 4:
    plt.yscale('log')
    p = float(input("Choose a value for probability: "))
    x = int(input("no. of successes / occurences: "))  
    
    if (p < 0) or (p > 1):
        sys.exit("Error as probability cant be outside of 0 and 1.")
        
    geom_cdf = 0
    k = 1
    
    if pdfcdf == True:
        while k <= x:
            geom_pdf = p*(1-p)**(k-1)
            geom_cdf += geom_pdf
            print(f"Successes: ({k}) | PDF: {geom_pdf:>12.6g} | Running CDF: {geom_cdf:>12.6g}")
            y.append(geom_pdf)
            X_vals.append(k)
            k += 1
        print(f"Final CDF is:  {geom_cdf:>17.11g}")
    else:
        geom_pdf = p*(1-p)**(x-1)
        print(f"PDF at point {x} is {geom_pdf:>17.15g}")

# Negative Binomial
elif choice == 5:
    plt.yscale('log')
    p = float(input("Choose a value for probability: "))
    r = int(input("Testing up until the rth success: "))
    x = int(input(f"no. of trials up to the {r}th success: "))  
    negbi_cdf = 0
    k = 1
    
    if r > x or (p > 1 or p < 0):
        sys.exit("Invalid: successes cannot be greater than the number of trials and probability of successes must be in the range 0 to 1.")
        
    if pdfcdf == True:
        while k <= x:
            while k <= r:  
                negbi_pdf = 0
                k += 1
            
            
            if k > x:
                break
                
            negbi_pdf = (factorial(k-1)/(factorial(r-1)*factorial(k-r)))*(p**r)*(1-p)**(k-r)
            negbi_cdf += negbi_pdf
            y.append(negbi_pdf)
            X_vals.append(k)
            print(f"Successes: ({k}) | PDF: {negbi_pdf:>12.6g} | Running CDF: {negbi_cdf:>12.6g}")
            k += 1
            
        print(f"Final CDF is:  {negbi_cdf:>24.22g}")
        print(f"number of successes is {r} and the number of trials is {x}")
    else:
        negbi_pdf = (factorial(x-1)/(factorial(r-1)*factorial(x-r)))*(p**r)*(1-p)**(x-r)
        print(f"PDF at point {x} is {negbi_pdf:>17.15g}")

if pdfcdf == True:
    fig, ax = plt.subplots(figsize=(8, 5))
    
    X = np.array(X_vals) 
    y = np.array(y)
    
    ax.plot(X, y, color="blue", linewidth=0.8)
    ax.set_snap(True)
    
    fill_start = a if choice == 2 else 0
    plt.fill_between(X, y, 0, where=(X > fill_start), color='g')
    
    cursor = matplotlib.widgets.Cursor(ax, useblit=True, horizOn=True, vertOn=True, color="red")
    ax.set_title("Distribution plot")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True)
    plt.show()
