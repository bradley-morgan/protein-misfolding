import subprocess


def run():
    out = subprocess.run(['!pip install colormap'])
    print("The exit code was: %d" % out.returncode)