import gen1
import gen2
import gen3
import train_part1
import train_part2
import train_part3
import os
import zipfile
import p1
import p2
import p3
import filter
import time


def zipDir(dirpath, outFullName):
    zip = zipfile.ZipFile(outFullName, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(dirpath):
        fpath = path.replace(dirpath, '')
        for filename in filenames:
            zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
    zip.close()


if '__main__' == __name__:
    all_begin = time.time()
    gen1.gen()
    print('time_is', time.time() - all_begin)
    gen2.gen()
    print('time_is', time.time() - all_begin)
    gen3.gen()
    print('time_is', time.time() - all_begin)
    train_part1.train()
    train_part2.train()
    train_part3.train()
    p1.eval()
    p2.eval()
    p3.eval()
    filter.work()
    all_end = time.time()
    print(all_end - all_begin)
    zip_path = '../temp_data/data'
    zipDir(zip_path, '../result/data.zip')
