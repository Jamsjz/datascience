with open("name.txt", "r") as f:
    names_file = f.read()


def search_name(target_start):

    start_of_name = -1
    names = set()
    target_end = " "

    if target_start.isalpha():
        for i, char in enumerate(names_file):
            if char == target_start.upper():
                start_of_name = i
            if char == target_end and start_of_name != -1:
                name = names_file[start_of_name: i + 1]
                names.add(name)
                start_of_name = -1
    else:
        print("Enter a Alphabet:")
    print(*names)


print("Enter 'q' or 'quit' to exit.\n")




while True:
    start = input("Enter a alphabet: ")
    if start.lower() == "q" or start.lower() == 'quit':
        break
    if start.isalpha():
        search_name(start)
    else:
        print("Enter a valid alphabet")
