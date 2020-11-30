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

def accelerate():
    app('System Events').key_down('p')


# EXAMPLE USE
#  time.sleep(1)
#  app('sixtyforce').activate()
#  time.sleep(2)
#  accelerate()
#  left(1)
#  right(1)
#  left(1)
#  left(1)
#  right(1)

