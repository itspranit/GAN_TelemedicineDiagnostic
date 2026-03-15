import os
import zipfile
import shutil

dataset_name="kmader/skin-cancer-mnist-ham10000"
target_dir="data"

print(f"Downloading {dataset_name} from Kaggle...")
os.system(f"kaggle datasets download -d {dataset_name} -p {target_dir}")
zip_file=f"{target_dir}/{dataset_name.split('/')[1]}.zip"

print("Extracting 10,000 images...")
with zipfile.ZipFile(zip_file,'r') as zip_ref:
    zip_ref.extractall(target_dir)
    
print("Flattening directories for dataset.py...")
for root,dirs,files in os.walk(target_dir):
    for file in files:
        if file.endswith('.jpg'):
            source_path=os.path.join(root,file)
            dest_path=os.path.join(target_dir,file)
            if source_path!=dest_path:
                shutil.move(source_path,dest_path)

print("Cleaning up leftover files...")
os.remove(zip_file)
for item in os.listdir(target_dir):
    item_path=os.path.join(target_dir,item)
    if os.path.isdir(item_path):
        shutil.rmtree(item_path)

print("Dataset ready! Check your 'data' folder.")