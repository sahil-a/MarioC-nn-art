import time
from appscript import app, k
from pynput.keyboard import Listener, Key

filename = "key_log.txt"  # The file to write characters to


def on_press(key):
    f = open(filename, 'a')  # Open the file

    if hasattr(key, 'char'):  # Write the character pressed if available
        f.write(key.char)
    else:  # If anything else was pressed, write [<key_name>]
        f.write('[' + key.name + ']')

    f.close()  # Close the file

# def on_press(key):
#     print("Key pressed: {0}".format(key))

# def on_release(key):
#     print("Key released: {0}".format(key))


def right(seconds):
    app('System Events').key_down('d')
    time.sleep(seconds)
    app('System Events').key_up('d')

def left(seconds):
    app('System Events').key_down('a')
    time.sleep(seconds)
    app('System Events').key_up('a')

app('sixtyforce').activate()
time.sleep(10)
with Listener(on_press=on_press) as listener:  # Setup the listener
    listener.join()  # Join the thread to the main thread