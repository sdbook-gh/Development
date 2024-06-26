from setuptools import setup

package_name = 'sender_receiver'

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
    maintainer='shenda',
    maintainer_email='shenda@china-aicc.cn',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sender = sender_receiver.sender:main',
            'receiver = sender_receiver.receiver:main',
        ],
    },
)
