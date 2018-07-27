from time import time


str_time = time()



print("star:%.3f"%str_time)

for i in range(1, 100000000):
    pass


stop_time = time()

print("stop:%.3f"%stop_time)



print("time:%.3f"%(stop_time - str_time))

