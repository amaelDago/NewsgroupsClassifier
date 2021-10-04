import gdown
import os
#import zipfile

#abs_path = os.path.abspath(os.path.dirname(__file__))

url = "https://drive.google.com/u/0/uc?id=1PPO1GysKCu_gJxbXd428q32o8ThOAJKE&export=download"
output = "dependencies.zip"
gdown.download(url, output)

#zip = os.path.join(abs_path, "dependencies.zip")

#zip = zipfile.ZipFile("zip")
print("Unzip ...")
os.system("apt-get update && apt-get install unzip && unzip -a dependencies.zip")

