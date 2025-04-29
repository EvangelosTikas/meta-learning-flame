# dataset_wrapper/loaders.py
import os
import requests
import tempfile

_CACHE = {}

def download_if_url(path_or_url):
    if path_or_url.startswith("http"):
        if path_or_url in _CACHE:
            return _CACHE[path_or_url]
        r = requests.get(path_or_url)
        r.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(r.content)
        tmp.close()
        _CACHE[path_or_url] = tmp.name
        return tmp.name
    else:
        return path_or_url
