from setuptools import setup, find_packages

setup(  
        name = 'mysummarizer',
        version = '1.0',
        packages = find_packages(exclude=['tests*']),
        license = 'MIT',
        description = 'Used to summarize a large body of text into smaller bits',
        long_description = open('README.md').read(),
        install_requires = ['numpy','transformers','torch','nltk'],
        url = "https://github.com/<gogzicole>/<summarizermodule>"
    )