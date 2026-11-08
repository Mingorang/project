import time 
import random
import numpy as np
import matplotlib.pyplot as plt
biggest = []
for a in range(0,1000):
    alpha = []
    for b in range(0,10):
        beta=[]
        upper=int(random.randrange(1,10**1,1))/(10**6)
        for c in range(0,10):
            start = time.time()


            k=0
            while (time.time()-start) < upper:
                k+=1
                pass
            beta.append(k)
            #print(f"Number reached in: {time.time()-start:.6} seconds is {k}")
            c+=1

        #print(sum(beta))
        #print(f"Mean numbers printed per second is {sum(beta)/upper}")
        alpha.append(sum(beta)/upper)

    #print(max(alpha))
    biggest.append(max(alpha))
print("")
print(max(biggest))
#Need to add matplot lib and difficult part will be introducing regressions, adding a list of reasonable methods similar to lines 9-25 of num_int.

#Just plotting atm
x = np.logspace(-3, 3, num=len(biggest))
y = np.array(biggest, dtype=float) / 1e6
plt.scatter(x, y)
plt.title("Big O notation")
plt.xlabel("time taken between iterations")
plt.ylabel("Number of iterations reached (Millions)")
plt.show()
#Will add regressions soon, this definitely looks like a log regression would work well.
