import urllib.request
import zipfile
import gzip
import tarfile
import bz2
import os

def url_download(url, name):
    urllib.request.urlretrieve(url, name)
    return


def extract_zip(location, filename):
    zip_file = zipfile.ZipFile(os.path.join(location, filename))
    for names in zip_file.namelist():
        zip_file.extract(names, location)
    zip_file.close()


def extract_gz(location, filename):
    f_name = filename.replace(".gz", "")
    g_file = gzip.GzipFile(os.path.join(location, filename))
    open(os.path.join(location, f_name), "wb+").write(g_file.read())
    g_file.close()

def extract_tar(location, filename):
    tar = tarfile.open(os.path.join(location, filename))
    names = tar.getnames()
    for name in names:
        tar.extract(name, location)
    tar.close()

def extract_bz2(location, filename):
    filepath = os.path.join(location, filename)
    zipfile = bz2.BZ2File(filepath)
    data = zipfile.read()
    newfilepath = filepath[:-4]
    open(newfilepath, 'wb').write(data)

extract_map = {
    'zip':extract_zip,
    'gz':extract_gz,
    'tar':extract_tar,
    'bz2':extract_bz2
}

def download_extract(url, cache_location, download_name, extract=None, extract_name='dfgfdg'):
    if not os.path.exists(os.path.join(cache_location, download_name)) and not os.path.exists(os.path.join(cache_location, extract_name)):
        print('Downloading '+download_name)
        url_download(url+download_name,
            os.path.join(cache_location, download_name))
        print(download_name+' downloaded')

    if not os.path.exists(os.path.join(cache_location, extract_name)):
        if extract is not None:
            print('extracting '+download_name)
            extract_map[extract](cache_location, download_name)
            print(download_name+' extracted')