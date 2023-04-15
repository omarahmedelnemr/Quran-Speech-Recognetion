import subprocess
import os



#Some Functions
def clear_folder(path):
    """
    Recursively remove all files and directories within a folder.
    """
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
def clear_console():
    if os.name == "nt":
        os.system("cls")  # For Windows
    else:
        os.system("clear")  # For Unix/Linux/MacOS
clear_console()

if input("Clear All Folders?(Y/N) ").lower() =='y':
    print("Clearing The Folders")

    clear_folder("./Data")
    clear_folder("./PreprossedData")
    clear_folder("./Spectro")
    clear_folder("./TestSpectro")

    print("Folders are Clear")

if input("Start Scripts? (Y/N)").lower() == 'y':
    clear_console()

    scriptsList = ['Convert.py',"Preprocces.py",'spectrogram.py']

    for script in scriptsList:

        print("{} is Running ...".format(script))

        subprocess.run(["python3", script])

        print("{} is Done !!".format(script))

        clear_console()


    print("All Scripts Done !")

if input("Start Model Training? (Y/N) ").lower() == 'y':
    clear_console()
    subprocess.run(["python3", 'cnn 2.py'])

print("All Done !")