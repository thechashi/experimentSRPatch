"""
Downloader module to fetch the image zip files from the toml
"""
import os
import tarfile
import zipfile
from pathlib import Path


def get_file_name(url):
    """

    :param url: file url
    :return: file_name, file_ext, odir
    """
    base_name = os.path.basename(url)
    base_name = str.split(base_name, sep="?")[0]
    file_name, file_ext = (
        str.split(base_name, sep=".")[0],
        ".".join(str.split(base_name, sep=".")[1:]),
    )

    return file_name, file_ext


def file_extraction(file_tmp, file_ext, odir):
    """

    :param file_tmp: the downloaded file
    :param file_ext: the extension of the file
    :param odir: the output directory of the file
    :return:
    """
    odir = os.path.abspath(odir)
    if not os.path.isdir(odir):
        os.makedirs(odir)
    if file_ext == "zip":
        with zipfile.ZipFile(file_tmp) as zip_ref:
            zip_ref.extractall(odir)
    if file_ext in ("tar", "tar.gz", "tar.xz"):
        tar = tarfile.open(file_tmp)
        tar.extractall(odir)


def data_download(datadict, file_down):
    """

    :param datadict: the dictionary that contains the download links
    :param odir: the path where the files will be extracted
    :return:
    """
    down_path = Path(file_down)
    if not os.path.isdir(down_path):
        os.mkdir(down_path)

    for _, url_key in enumerate(datadict):
        url = datadict[url_key][0]
        odir = datadict[url_key][1]
        file_name, file_ext = get_file_name(url)
        file_down_path = down_path / (file_name + "." + file_ext)
        if not file_down_path.exists():
            os.system(f"wget {url} -O {file_down_path}")
            file_extraction(file_down_path, file_ext, odir)
        if url_key == "Medical_3D":
            files = os.listdir(odir / "Medical_3D")
            for compressed_file in files:
                file_extraction(odir / "Medical_3D" / compressed_file, file_ext, odir)
                os.remove(odir / "Medical_3D" / compressed_file)
