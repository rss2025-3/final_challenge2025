
from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'final_challenge2025'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.xml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python',
        'scipy',
        'matplotlib'
    ],
    zip_safe=True,
    maintainer='racecar',
    maintainer_email='fionaw5th@gmail.com',
    description='Final Challenge 2025 ROS 2 nodes and tools',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "trajectory_planner = final_challenge2025.trajectory_planner:main",
            "basement_point_publisher = final_challenge2025.basement_point_publisher:main",
            "trajectory_follower = final_challenge2025.trajectory_follower:main",
            "a_star_final = final_challenge2025.a_star_final:main",
            "color_segmentation = final_challenge2025.color_segmentation:main",
            "heist_stopping = final_challenge2025.heist_stopping:main",
            "detection_node = final_challenge2025.model.detection_node:main",
            "parking_controller = final_challenge2025.parking_controller:main"
        ],
    },
)
