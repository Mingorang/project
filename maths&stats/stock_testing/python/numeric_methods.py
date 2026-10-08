import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import simpson as simps
from scipy.integrate import quad
import sys

#Define a reasonable function to be used in the integral definition, so functions can only be defined with the following expressions 
#this list is not exhaustive and the integral still works despite operands not being included
def reasonable_function(expression):
    allowed = {
        "exp": np.exp,
        "e": np.e,
        "pi": np.pi,
        "sqrt": np.sqrt,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "csc": np.asin,
        "sec": np.acos,
        "cot": np.atan,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "log": np.log,
        "log10": np.log10,
    }
    def f(x):
        return eval(expression, {**allowed}, {"x": x})
    return f
#This function is used to create a sample field, this sample only takes finite values, so (1/0) would not get here.    
def sample_function(f, a, b, dx):

    x = np.arange(a, b + dx, dx)

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        y = f(x)

    valid = np.isfinite(y)

    x = x[valid]
    y = y[valid]

    return x, y

#The previous function relates to this one; we take percentiles and use a user-defined variable called alpha to 'prune' values considered to be too large or divergent, avoiding ZeroDivision errors and getting a numerical value after integration.
def detect_singularities(x, y, alpha):
    lower = np.percentile(y, 100 - alpha)
    upper = np.percentile(y, alpha)
    outliers = (y < lower) | (y > upper)
    gradient = np.abs(np.diff(y))
    gradient_limit = np.percentile(gradient, alpha)
    regions = []
    inside = False
    for i in range(len(gradient)):
        if outliers[i] and gradient[i] > gradient_limit:
            if not inside:
                start = i
                inside = True
        else:
            if inside:
                end = i
                regions.append((x[start], x[end]))
                inside = False
    if inside:
        regions.append((x[start], x[-1]))
    return regions
#Splits the integral and ignores values approaching (+/-)infinity. For example,  integrating 1/(x-5) from x-->10 will return nan, this can be reduced (with some error) to integrating from 0-->4.999 + 5.001-->10
def split_intervals(a, b, regions, dx):
    intervals = []
    current = a
    gap = 5 * dx
    for left, right in regions:
        if left - gap > current:
            intervals.append((current, left - gap))
        current = right + gap
    if current < b:
        intervals.append((current, b))
    return intervals
#allows for writing accurate limits
import ast
import math
import operator

def string_to_float(text):
    
    cleaned = text.replace(" ", "").lower()
    
    
    cleaned = cleaned.replace("pi", "*pi").replace("π", "*pi")
    if cleaned.startswith("*"):
        cleaned = cleaned[1:]
    cleaned = cleaned.replace("**pi", "*pi") # Fix double asterisks if user wrote "2*pi"
    
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg
    }
    
    tree = ast.parse(cleaned, mode='eval')
    
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant):
            return float(node.value)
        elif isinstance(node, ast.Name):
            if node.id == "pi":
                return math.pi
            raise ValueError(f"Unsupported variable: {node.id}")
        elif isinstance(node, ast.BinOp):
            return allowed_operators[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            return allowed_operators[type(node.op)](_eval(node.operand))
        else:
            raise TypeError(f"Unsupported syntax: {type(node).__name__}")
            
    return _eval(tree)

# Examples
   

#Actual integration calculator
def main():
       #Initialising variables
        print("Approximating integrals with rectangles")    
        expr = input("Integral as a function of x: ")
        a = string_to_float(input("lower limit: "))
        print(a)
        b= string_to_float(input("upper limit: "))
        print(b)
        dx = float(abs(b-a)/1e4)
        #larger alpha consideres less points as divergent
        alpha = float(input("measure for finding singularities: (85<=alpha<100): "))
        if alpha<85 or alpha>=100:
            sys.exit("You know this value isnt allowed.")
        #Riemann sum (Method of integration using rectangles)
        f = reasonable_function(expr)
        x_rect, y_rect = sample_function(f,a,b,dx)
        area = np.sum(y_rect * dx)
        
        
        #calculating % error(s) and simpson function
        x, y = sample_function(f,a,b,dx)
        regions = detect_singularities(x,y,alpha)
        intervals = split_intervals(a,b,regions,dx)
        #actual function
        for left, right in intervals:
          x_plot, y_plot = sample_function(f, left, right, dx)
         
        actual_area = 0
        for left, right in intervals:
          actual_area += quad(f,left,right)[0]
        #Simpsons (Method of integration w/ quadratics)
        sim = simps(y,x)
        error = 100*(abs(sim-area)/sim)
        error_simpsons = 100*((abs(sim-actual_area))/actual_area)
        error_riemann = 100*((abs(actual_area-area))/actual_area)

     

        #Plotting results
        plt.figure(figsize=(10,6))
        plt.plot(x_plot,y_plot, color="red", label=f"y = {expr}")
        plt.bar(x_rect,y_rect,color="lightblue",edgecolor="black",  label="Rectangles", width = dx, align="edge")
        plt.axhline(0,color="black", linewidth=0.1)
        plt.title("Integral approximation with rectangles")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(True)
        #Analysis of whether Simpson's or Riemann's integration technique is more accurate.
        print(f"Simpson's rule gave an integral of {sim}, Riemann's sum from {a} to {b} gives {area.__round__(11)}, so the % error between the two approximations  is: {error}% ")
        print("")
        print(f"However, the actual integral is: {actual_area.__round__(9)}")
        print("")
        if error_simpsons > 100 or error_riemann>100 :
            error_simpsons = 100
            error_riemann = 100
        print(f"Riemann's is {100-error_riemann}% accurate,  Simpson's is {100-error_simpsons}% accurate")
        if error_simpsons < error_riemann:
             print("Simpson's rule give a more accurate answer")
        elif error_riemann < error_simpsons:
             print("Riemann's is more accurate,")
        else:
             print("No value to compare to.")
             print("For this to be outputted function is most likely divergent.")
        time.sleep(3)
        plt.draw()
        plt.pause(600)
#Should add graphing option for simpson's method
#Since file is numeric methods, should cover things other than integration.

if __name__ == "__main__":
    main()
#Maybe add simpson's approximation to plotting
