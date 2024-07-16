
import os
from colorama import Fore, Style, init


def print_directory_structure(startpath, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = []

    for root, dirs, files in os.walk(startpath):
        # Calculate the level of depth
        level = root.replace(startpath, '').count(os.sep)

        # Skip directories that are in the exclude list
        if any(excluded in root for excluded in exclude_dirs):
            continue

        indent = ' ' * 4 * (level)
        print(f"{Fore.BLUE}{Style.BRIGHT}{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)

        # for f in files:
        #     print(f"{Fore.RED}{Style.NORMAL}{subindent}{f}")

        # Remove excluded directories from the list of dirs to walk
        dirs[:] = [d for d in dirs if os.path.join(root, d) not in exclude_dirs]


if __name__ == '__main__':
    # Initialize colorama
    init(autoreset=True)
    # Specificați calea directorului de bază
    base_path = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\MODALITIES'

    # Lista de directoare de exclus (relativ la directorul de bază)
    exclude_dirs = ['exported', '.venv', 'output', 'saveroot', '.git', '.idea' ]

    # Afișăm structura directorului
    print_directory_structure(base_path, exclude_dirs)
