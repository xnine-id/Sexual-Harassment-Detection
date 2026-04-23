import gdown

def download_model(file_id, destination_path):
    gdown.download(f"https://drive.google.com/uc?export=download&id={file_id}", destination_path, quiet=False)