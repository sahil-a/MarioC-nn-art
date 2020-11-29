import time
from appscript import app, k

def right(seconds):
    app('System Events').key_down('d')
    time.sleep(seconds)
    app('System Events').key_up('d')

def left(seconds):
    app('System Events').key_down('a')
    time.sleep(seconds)
    app('System Events').key_up('a')

# EXAMPLE USE
#  app('sixtyforce').activate()
#  time.sleep(2)
#  right(2)
#  left(2)
