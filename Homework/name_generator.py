import random  # importing random std library to generate random number


# function to generate random number
def random_number_generatorpi(total_number_of_name):
    random_index = random.randint(0, total_number_of_name)
    return random_index


# function that search and print name starting from the user given alphabet
def search_name(target_start, names_file):
    start_of_name = -1
    names = []
    target_end = " "

    if target_start.isalpha():
        # it goes through name.txt file and iterate every single character
        for i, char in enumerate(names_file):
            if char == target_start.upper():
                start_of_name = i  # Assign the index of the names starting from user given alphabet
            if char == target_end and start_of_name != -1:
                name = names_file[start_of_name: i + 1]
                names.append(name)
                start_of_name = -1
    else:
        print("Enter an alphabet:")

    numbers_of_name = len(names)
    
    print(names[random_number_generator(numbers_of_name - 1)])


def main():
    with open("name.txt", "r") as f:
        names_file = f.read()

    print(" \nEnter 'quit' to exit.\n")

    while True:
        start = input("Enter an alphabet: ")
        if start.lower() == 'quit':
            break
        if start.isalpha() and len(start) == 1:
            search_name(start, names_file)
        else:
            print("Enter a valid alphabet")


if __name__ == "__main__":
    main()
