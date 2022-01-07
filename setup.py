from setuptools import setup, find_packages
from setuptools.command.install import install
import os, warnings

def parse(reqstr):
    filename = getattr(reqstr, 'name', None)

    # Python 3.x only
    if not isinstance(reqstr, str):
        reqstr = reqstr.read()

    lines = []

    for line in reqstr.splitlines():
        line = line.strip()
        if line == '':
            continue
        elif not line or line.startswith('#'):
            # comments are lines that start with # only
            continue
        elif line.startswith('git+'):
            warnings.warn('git repos not supported (im pissed as you are). Skipping.')
            continue
        elif line.startswith('-r') or line.startswith('--requirement'):
            _, new_filename = line.split()
            new_file_path = os.path.join(os.path.dirname(filename or '.'),
                                         new_filename)
            with open(new_file_path) as f:
                for requirement in parse(f):
                    lines.append(requirement)
        elif line.startswith('-f') or line.startswith('--find-links') or \
                line.startswith('-i') or line.startswith('--index-url') or \
                line.startswith('--extra-index-url') or \
                line.startswith('--no-index'):
            warnings.warn('Private repos not supported. Skipping.')
            continue
        elif line.startswith('-Z') or line.startswith('--always-unzip'):
            warnings.warn('Unused option --always-unzip. Skipping.')
            continue
        else:
            lines.append(line)

    return lines

requirementPath =  'requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = parse(f.read())

class InstallPixray(install):
    def run(self):
        install.run(self)
        # path = os.getcwd().replace(" ", "\ ").replace("(","\(").replace(")","\)") + "/bin/"
        os.system("git clone https://github.com/pixray/diffvg && cd diffvg && git submodule update --init --recursive && DIFFVG_CUDA=1 python setup.py install")
        os.system("git clone --recursive https://github.com/dazhizhong/v-diffusion-pytorch ")
        # os.system("cp -r v-diffusion-pytorch/diffusion pixray/.")
        print("installing git dependencies")
        os.system(f"pip install -r {requirementPath}")

setup(
    name='pixray',
    version='1.5.1',    
    description='text-to-image neural generation engine',
    url='https://github.com/dazhizhong/pixray',
    author='Dazhi Zhong, dribnet',
    packages=["."]+find_packages(),
    install_requires=install_requires,
    classifiers=[],
    cmdclass={'install': InstallPixray},
)