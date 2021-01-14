from setuptools import setup

package_name = 'people_detection_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yukiho-YOSHIEDA',
    maintainer_email='is0436er@ed.ritsumei.ac.jp',
    description='人物の推定をするROS2パッケージ',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'people_detection_ros2 = people_detection_ros2.people_detection_node:main',
        ],
    },
)
