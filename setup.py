from setuptools import setup
from setuptools.command.install import install
import os, warnings

class InstallPixray(install):
    def run(self):
        install.run(self)
        # path = os.getcwd().replace(" ", "\ ").replace("(","\(").replace(")","\)") + "/bin/"
        os.system("git clone https://github.com/pixray/diffvg && cd diffvg && git submodule update --init --recursive && DIFFVG_CUDA=1 python setup.py install")
        os.system("git clone https://github.com/dazhizhong/v-diffusion-pytorch -b addcolabs")

setup(
    name='pixray',
    version='1.5.1',    
    description='text-to-image neural generation engine',
    url='https://github.com/dazhizhong/pixray',
    author='Dazhi Zhong, dribnet',
    packages=['.'],
    install_requires=[
        'braceexpand',
        'kornia',
        'omegaconf'
    ],
    classifiers=[],
    cmdclass={'install': InstallPixray},
)