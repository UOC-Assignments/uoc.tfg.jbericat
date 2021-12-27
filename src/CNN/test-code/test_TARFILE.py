
import tarfile
import os.path

MODEL_PATH = "bin/CNN/cnn-training_20211226-104327" 

archive = tarfile.open(MODEL_PATH+".tar.gz", "w|gz")
archive.add(MODEL_PATH, arcname="")
archive.close()

print("\nTraining results file -> " + os.path.abspath(MODEL_PATH) + ".tar.gz\n")