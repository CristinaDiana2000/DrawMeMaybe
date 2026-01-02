from setuptools import find_packages, setup

package_name = 'my_sample_pkg_python'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='drawmemaybe',
    maintainer_email='cristina.tutunariu@tha.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'test_node = my_sample_pkg_python.test_node:main',
            'move_to_home = my_sample_pkg_python.move_to_home:main',
            'move_to_home_traj = my_sample_pkg_python.move_to_home_traj:main',
            'haus_des_nikolaus = my_sample_pkg_python.haus_des_nikolaus:main',
            'test = my_sample_pkg_python.test:main',
            'move_moveit = my_sample_pkg_python.move_moveit:main',
            'move_moveit_svg1 = my_sample_pkg_python.move_moveit_svg1:main',
            'move_moveit_svg2 = my_sample_pkg_python.move_moveit_svg2:main',
        ],
    },
)
