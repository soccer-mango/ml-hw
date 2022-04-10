"""
This file is to check if the folder that the student is trying to submit
is of this exact structure (with exact naming):
| code
    | img_comression.py
    | kmeans.py
    | song_clustering.py
| data
    | <we don't care about what goes on here. we'll test it using our version of the data>
| writeup.md

And then it will zip

This code is inspired by the submission zip file created by Professor
James Tompkin (james_topkin@brown.edu) for CSCI 1430 - Brown University.

Adapted by Nam Do (nam_do@brown.edu) - Spring 2021.
"""

import sys
import os, zipfile

########################################## HELPER FUNCTIONS ##########################################

# Function from https://stackoverflow.com/questions/1724693/find-a-file-in-python
def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


# Function adapted from https://www.geeksforgeeks.org/working-zip-files-python/
def get_all_file_paths(directory, ext):
    # Initializing empty file paths list
    file_paths = []

    # Crawling through directory and subdirectories
    for root, directories, files in os.walk(directory):
        for filename in files:
            if filename.endswith(ext):
                # Join the two strings in order to form the full filepath.
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)

    # returning all file paths
    return file_paths


########################################## MAIN ZIP LOGIC ##########################################
def main():
    # First, check what directory that we're running this from
    curdir = os.getcwd()
    failed = False
    # If the current directory doesn't contain this script, we'll exit and
    # tell the student to chdir to the right directory
    if find('zip_assignment.py', curdir) == None:
        # We haven't found this file, and so we will print out a message and sys exit
        print("We cannot find the file zip_assignment.py in the directory that you are")
        print("executing this script from. Please use command 'cd <path>' to change to the right")
        print("directory that contains zip_assignment.py and execute this script again.")
        sys.exit()

    # Check if there are the right files
    writeup_path = "writeup.md"
    if not os.path.exists(writeup_path):
        failed = True
        print("Issue: This directory does not contain '{}'.".format(writeup_path))
        print("Please make sure that your written solution is a pdf file and named '{}' and in the same directory as 'zip_assignment.py'.".format(writeup_path))
        print("\n")
    # Check if there is a folder named code
    if not os.path.exists("code"):
        failed = True
        print("Issue: This directory does not contain a subdirectory 'code'.")
        print("Please make sure that all your code is located in the subdirectory 'code' and that code is in the same directory as 'zip_assignment.py'")
        print("\n")


    # Check if there exists all the .sql files that we're looking for
    python_files = ["kmeans.py", "song_clustering.py", "img_compression.py"]

    # Check if there is the assignment.py file that we're looking for
    for pf in python_files:
        path = "code" + os.sep + pf
        if not os.path.exists(path):
            failed = True
            print("Issue: File not found: {}.".format(path))
            print("Please make sure your '{}' file is put in the subdirectory 'code'.".format(str(pf)))


    if failed:
        print("Please fix all your issues for zipping code to proceed.")
        sys.exit()

    print("Writting into zip file...")
    zip_path = "ml-submission-1951A.zip"
    with zipfile.ZipFile(zip_path, "w") as zip:
        # Write the written file
        if os.path.exists(writeup_path):
            zip.write(writeup_path)

        # Write the .py files
        for f in get_all_file_paths("code" + os.sep, ".py"):
            zip.write(f)


    print("Done! Wrote the submission zip to {}".format(zip_path))






if __name__ == "__main__":
    main()
