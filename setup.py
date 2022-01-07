from setuptools import setup

from setuptools import setup
from setuptools.command.install import install
import os

class InstallPixray(install):
    def run(self):
        install.run(self)
        # path = os.getcwd().replace(" ", "\ ").replace("(","\(").replace(")","\)") + "/bin/"
        os.system("git clone https://github.com/pixray/diffvg && cd diffvg && git submodule update --init --recursive && CMAKE_PREFIX_PATH=$(pyenv prefix) DIFFVG_CUDA=1 python setup.py install")
        os.system("git clone https://github.com/pixray/v-diffusion-pytorch")

setup(
    name='pixray',
    version='1.5.1',    
    description='neural generation engine',
    url='https://github.com/dazhizhong/pixray',
    author='Dazhi Zhong, dribnet',
    packages=['pixray'],
    install_requires=[],
    classifiers=[],
    cmdclass={'install': InstallPixray},
)