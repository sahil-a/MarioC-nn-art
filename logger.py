import time
import sys
import os
from appscript import app, k
from pynput.keyboard import Listener, Key

sum = 0
iter = 0

def logger(track, trial):
    app('sixtyforce').activate()
    time.sleep(10)
    base = track + "_trial_" + trial
    f = open("%s" % base, "a")

    start = time.time()
    iter1 = 0
    keyval = 0
    while(keyval != -3):
        path = "/data/%s/%s_%03d" % (base, base, iter1)
        screenshot(path)
        keyval = keystroke()
        # if(keyval == -3):
        #     break
        f.write("%s, %3f \n" % (path, keyval))
        iter1 = iter1 + 1

    end = time.time()
    print("Total time taken: %3f sec" % ((end-start)/iter))


def screenshot(path):
    # Code for taking screenshot and save to path
    os.system(f'screencapture -R 750,245,620,450 {path}')

def keystroke():
    global iter 
    global sum 
    with Listener(on_press=on_press) as listener:  # Setup the listener
        listener.join()  # Join the thread to the main thread
    sum = 0
    iter = 0
    time.sleep(1)
    # elap = 0
    # start = time.time()
    # while(elap < 1):
    #     elap = time.time() - start
    #     # val = code for logging keystroke (1 for right, -1 for left, 0 if nothing, -3 if q)
    #     if(val == -3):
    #         return -3
    #     val = 1 # delete
    #     sum = sum + val
    #     iter = iter + 1
    listener.stop()
    return sum / iter

def on_press(key):
    global iter 
    global sum 
    val = 0 
    iter = iter + 1 
    if hasattr(key, 'char'):  # Write the character pressed if available
        if key.char == 'd':
            val = 1
        elif key.char == 'a':
            val = -1
    sum = sum+ val 
    # # elif key.name == 'a':
    # #     print("a")
    # # elif key.name == 's':
    # #     print("s")
    # # elif key.name == 'd':
    # #     print("d")
    # else:  # If anything else was pressed, write [<key_name>]
    #     f.write('[' + key.name + ']')

    # f.close()  # Close the file
# First command line argument: track, second command line argument: trial number
# Use q to break from program
if __name__ == "__main__":
    track = sys.argv[1]
    trial = sys.argv[2]
    logger(track, trial)

