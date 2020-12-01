from test_part1 import *
from test_part2 import *
import zipfile
import os
import shutil
def zip_dir(dirname,zipfilename):
  filelist = []
  if os.path.isfile(dirname):
    filelist.append(dirname)
  else :
    for root, dirs, files in os.walk(dirname):
      for name in files:
        filelist.append(os.path.join(root, name))
  zf = zipfile.ZipFile(zipfilename, "w", zipfile.zlib.DEFLATED)
  for tar in filelist:
    arcname = tar[len(dirname):]
    #print arcname
    zf.write(tar,arcname)
  zf.close()
if __name__=="__main__":
    test_part1()
    test_part2()
    zip_dir(r'../temp_data/result/',r'../result/data.zip')
    #shutil.rmtree(r'../result/data')
