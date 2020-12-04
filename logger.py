import time
import sys
import os
from appscript import app, k
from pynput.keyboard import Listener, Key
from threading import Thread
from pathlib import Path

sum = 0


def logger(track, trial):
    app('sixtyforce').activate()
    time.sleep(10)
    base = "data/data/" + track + "_trial_" + trial

    f = open("%s" % base, "a")

    start = time.time()
    iter1 = 0
    keyval = 0
    while (True):
        path = "data/screenshots/%s_%05d.png" % (base, iter1)
        screenshot(path)
        keyval = keystroke()
        if keyval == -3:
            break
        f.write("%s, %3f \n" % (path, keyval))
        iter1 = iter1 + 1

    end = time.time()
    f.close()
    print("Total time taken: %3f sec" % (end-start))


def screenshot(path):
    # Code for taking screenshot and save to path
    os.system(f'screencapture -R 750,245,620,450 {path}')
    print("after")

def keystroke():
    global sum
    sum = 0
    with Listener(on_press=on_press) as listener:
        def time_out(period_sec: int):
            time.sleep(period_sec)
            listener.stop()
            # Setup the listener
        Thread(target=time_out, args=(0.5,)).start()
        listener.join()  # Join the thread to the main thread
    return sum / 15


def on_press(key):
    global iter
    global sum
    val = 0
    if hasattr(key, 'char'):  # Write the character pressed if available
        if key.char == 'd':
            val = 1
        elif key.char == 'a':
            val = -1
        elif key.char == 'q':
            val = 0
            sum = -45
    sum = sum + val


# First command line argument: track, second command line argument: trial number
# Use q to break from program
if __name__ == "__main__":
    track = sys.argv[1]
    trial = sys.argv[2]
    logger(track, trial)
