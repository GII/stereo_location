import os
from setuptools import find_packages, setup
from glob import glob

package_name = 'stereo_location'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name, 'config'), glob('config/*.[yma]*')),
        (os.path.join('share', package_name, 'config'), glob('config/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'stereo_location = stereo_location.stereo_location:main',
            'object_location = stereo_location.object_location:main',
            'robot_location = stereo_location.robot_location:main',
        ],
    },
)
