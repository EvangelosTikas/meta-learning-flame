import os
import requests
import tempfile

_CACHE = {}
def download_if_url(path_or_url):
    if path_or_url.startswith("http"):
        if path_or_url in _CACHE:
            return _CACHE[path_or_url]


        response = requests.get(path_or_url)
        response.raise_for_status()
        suffix = os.path.splitext(path_or_url)[-1] or ".bin"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(response.content)
        tmp.close()

        _CACHE[path_or_url] = tmp.name
        return tmp.name
    return path_or_url
