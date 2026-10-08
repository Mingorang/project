import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
#Automated y-axis scale
def lin_log_auto_scale(y_data_list, threshold=100):
 
    if not isinstance(y_data_list, list):
        y_data_list = [y_data_list]
    
    all_data = np.concatenate([np.atleast_1d(data) for data in y_data_list])
    positive_y = all_data[all_data > 0]
    
    if len(positive_y) > 0:
        magnitude_span = np.max(positive_y) / np.min(positive_y)
        if magnitude_span > threshold:
            plt.yscale('log')
        else:
            plt.yscale('linear')
    else:
        plt.yscale('linear')

#def lin_log_auto_scale(y_sol,threshold):
#    if choice == '1':
#        def dydx(x,y):
#            return x+y
#        v0=0
#        x=np.linspace(a, b,1000)
#
#        sol_m1 = odeint(dydx, y0=v0, t=x, tfirst=True)
#        sol_m2 = solve_ivp(dydx, t_span=(-10,max(x)), y0=[v0], t_eval=x)
#        v_sol_m1 = sol_m1.T[0]
#        v_sol_m2 = sol_m2.y[0]
#        
#        positivey = y_sol[y_sol >0]
#        if len(positivey) > 0:
#                magnitude_span = np.max(positivey) / np.min(positivey)
#                if magnitude_span > threshold:
#                    plt.yscale('log')
#                else:
#                    plt.yscale('linear')
#        else:
#            plt.yscale('linear')
#
#User inputs
a = int(input("Lower limit of x/t: "))
b = int(input("Upper limit of x/t: "))
choice = input("Simple ODE or coupled ODE? (1/2): ")
#Single ODE option
if choice == '1':
    def dydx(x, y):
        return x + y
        
    v0 = 0
    x = np.linspace(a, b, 1000)

    sol_m1 = odeint(dydx, y0=v0, t=x, tfirst=True)
    sol_m2 = solve_ivp(dydx, t_span=(-10, max(x)), y0=[v0], t_eval=x)
    v_sol_m1 = sol_m1.T[0]
    v_sol_m2 = sol_m2.y[0]

    plt.plot(x, v_sol_m1, label="odeint solution")
    plt.plot(x, v_sol_m2, '--', label="solve_ivp solution")
    
    lin_log_auto_scale([v_sol_m1, v_sol_m2], threshold=30)

    plt.legend()
    plt.ylabel('$f(x)$')
    plt.xlabel('$x$')
    plt.show()
#Coupled ode option
elif choice == '2':
    def dS_dt(t, S):
        x, y = S
        dx_dt = 5*x - 70*y 
        dy_dt = x - y
        return [dx_dt, dy_dt]

    S0 = [1.5, 0.5]   
    t = np.linspace(a, b, 1000)

    solveivp_input = input("Do you want to use solve_ivp or odeint? (T/F): ").strip().upper()
    solveivp = solveivp_input in ['T', 'TRUE']
    
    if solveivp:
        solution = solve_ivp(dS_dt, t_span=(a, b), y0=S0, t_eval=t)
        x_sol, y_sol = solution.y
    else:
        odeintsolution_coupled = odeint(dS_dt, y0=S0, t=t, tfirst=True)
        x_sol = odeintsolution_coupled.T[0] 
        y_sol = odeintsolution_coupled.T[1]
#plots
    plt.plot(t, x_sol, label="$x(t)$")
    plt.plot(t, y_sol, label="$y(t)$")
    
    lin_log_auto_scale([x_sol, y_sol], threshold=100)
    plt.legend()
    plt.title("Coupled First-Order ODEs")
    plt.xlabel("t")
    plt.show()
