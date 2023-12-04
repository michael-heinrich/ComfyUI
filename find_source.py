# recursively find all python source files in the current directory

import os
import sys


def list_sources() -> list:
    '''
    function that recursively finds all python source files in the current directory
    and returns a list of their relative paths
    '''
    source_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                source_files.append(os.path.join(root, file))
    return source_files


def contains_substring(substring: str, file: str) -> bool:
    '''
    function that returns True if the given file contains the given substring
    '''

    try:
        with open(file, 'r') as f:
            for line in f:
                if substring in line:
                    return True
    except UnicodeDecodeError:
        print(f"UnicodeDecodeError: {file}")
        return False

    return False


def find_any(substring: str) -> list:
    '''
    function that recursively finds all files in the current directory
    that contain the given substring and returns a list of their relative paths
    '''

    files = list_sources()

    # write a result file at ./result.txt
    # each line shows the path to an analyzed file. At the end of each line,
    # the number of occurrences of the substring is shown:
    # if the substring was 'SVD_img2vid_conditioning', the result file would look like:
    # e.g. ./find_source.py: 0
    #     ./find_source.py: 1 x 'SVD_img2vid_conditioning'
    # only files that contain the substring have the substring written out,
    # all just print : 0

    match_paths = []

    with open('result.txt', 'w') as f:
        for file in files:
            if contains_substring(substring, file):
                with open(file, 'r') as f2:
                    count = 0
                    for line in f2:
                        if substring in line:
                            count += 1
                    f.write(f"{file}: {count} x '{substring}'\n")
                    match_paths.append(file)
            else:
                f.write(f"{file}: 0\n")



    return match_paths


# print the list of files that contain the given substring
substring = "VideoLinearCFGGuidance"
files = find_any(substring)
