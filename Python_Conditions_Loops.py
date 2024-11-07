my_list = [2, 3, 5, 3, 7, 2, 5, 8, 5]
duplicates = {}

#Print the duplicate value
#Print the positions (indices) where the duplicate appears in the list
for index, item in enumerate(my_list):
    if my_list.count(item) > 1:
        temp_list = []
        if duplicates.get(item):
            temp_list.extend(duplicates[item])
        temp_list.append(index)
        duplicates[item] = temp_list
        if len(temp_list) == my_list.count(item):
            print(f"Duplicate {item} found at positions: {temp_list}")
