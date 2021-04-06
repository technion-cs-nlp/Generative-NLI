import multiprocessing
import time
import torch

def a_function(ret_value):
    for i in range(10**8):
        pass
    ret_value.value = 3.145678

def b_function(ret_value):
    for i in range(10**8):
        pass
    ret_value.value = 3.145678

start = time.time()
ret_value = multiprocessing.Value("d", 0.0, lock=False)
ret_value2 = multiprocessing.Value("d", 0.0, lock=False)

# reader_process = multiprocessing.Process(target=a_function, args=[ret_value])
# # reader_process2 = multiprocessing.Process(target=b_function, args=[ret_value2])
# reader_process.start()
# # reader_process2.start()

# reader_process.join()
# reader_process2.join()
a_function(ret_value)
b_function(ret_value2)

print(ret_value.value)
print(time.time()-start)