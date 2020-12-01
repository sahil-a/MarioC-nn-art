import time
import sys

def logger(track, trial):
    base = track + "_trial_" + trial
    f = open("%s" % base, "a")

    start = time.time()
    iter = 0
    while(iter < 10):
        path = "/data/%s/%s_%03d" % (base, base, iter)
        screenshot(path);
        keyval = keystroke();

        f.write("%s, %3f \n" % (path, keyval))
        iter = iter + 1

    end = time.time()
    print("Total time taken: %3f sec" % ((end-start)/iter))


def screenshot(path):
    # Code for taking screenshot and save to path
    return;

def keystroke():
    sum = 0
    iter = 0
    elap = 0
    start = time.time()
    while(elap < 1):
        elap = time.time() - start
        # val = code for logging keystroke (1 for right, -1 for left)
        val = 1 # delete
        sum = sum + val
        iter = iter + 1
    return sum / iter

# First command line argument: track, second command line argument: trial number
if __name__ == "__main__":
    track = sys.argv[1]
    trial = sys.argv[2]
    logger(track, trial)
