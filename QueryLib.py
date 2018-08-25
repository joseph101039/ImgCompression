import sys

def query_yes_no(question):

    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}

    while True:
        sys.stdout.write(question + " [y/n] ")
        choice = input().lower()
        if choice == '':
            return valid["yes"]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")