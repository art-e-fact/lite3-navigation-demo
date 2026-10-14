from pathlib import Path

from setuptools import find_packages, setup


package_name = 'simulation_package'
package_root = Path(__file__).parent.resolve()
share_dir = 'share/' + package_name


def _data_files(source_dir: str, install_dir: str) -> list[tuple[str, list[str]]]:
    source_root = package_root / source_dir
    if not source_root.exists():
        return []

    collected: list[tuple[str, list[str]]] = []
    for path in sorted(source_root.rglob('*')):
        if not path.is_file():
            continue
        relative_parent = path.parent.relative_to(source_root)
        destination = Path(install_dir) / relative_parent
        collected.append((str(destination), [str(path.relative_to(package_root))]))
    return collected

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        (share_dir, ['package.xml']),
    ] + _data_files('launch', share_dir + '/launch') + _data_files('assets', share_dir + '/assets') + _data_files('config', share_dir + '/config'),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='azazdeaz',
    maintainer_email='azazdeaz@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'start_simulation = simulation_package.start_simulation:main',
            'newton_simulation_ros2 = simulation_package.newton_backend.newton_simulation_ros2:main',
        ],
    },
)
