import os
from pathlib import Path
import logging

project_name="moneylion"

list_of_files=[
    '.github/workflows/.gitkeep',
    f'src/{project_name}/__init__.py',
    f'src/{project_name}/components/__init__.py',
    f'src/{project_name}/utils/__init__.py',
    f'src/{project_name}/utils/common.py',
    f'src/{project_name}/config/__init__.yaml',
    f'src/{project_name}/config/configuration.py',
    f'src/{project_name}/pipeline/__init__.py',
    f'src/{project_name}/entity/__init__py',
    f'src/{project_name}/entity/config_entity.py',
    f'src/{project_name}/constants/__init__.py',
    'yamls/config.yaml',
    'yamls/params.yaml',
    'yamls/schema.yaml',
    'main.py',
    'Dockerfile',
    'setup.py',
    'research/research.ipynb',
    'templates/index.html',
    'serving/helpers.py',
    'serving/__init__.py',
    'app.py'
]

for filepath in list_of_files:
    filepath=Path(filepath)
    filedir,filename=os.path.split(filepath)

    if filedir != '':
        os.makedirs(filedir, exist_ok=True)
        logging.info( f'Creating directory {filedir} for file: {filename}')

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,'w') as f:
            pass
            logging.info(f'Creating empty file: {filepath}')
    else: 
        logging.info(f'{filename} already exists')