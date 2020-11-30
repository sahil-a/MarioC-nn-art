import os

def screenshot(path):
    os.system(f'screencapture -R 750,245,620,450 {path}')

# EXAMPLE USE
screenshot('screenshots/lol.png')
