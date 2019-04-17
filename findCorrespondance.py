import numpy as np


def findCorrespondance(a, b):
    """To extract corrospondance between 2 images from given file

    Args:
        a (int): First Image Number
        b (int): Second Image Number

    Returns:
        list: List of matching points: R,G,B,x1,y1,x2,y2
    """
    database_path = "Data/"
    matching_list = []

    if (a == 1):
        with open(database_path + "matching1.txt") as f:
            line_no = 1
            # print("File opened")
            for line in f:
                if line_no == 1:
                    start_index = line.find(':')
#                     print("start_index = ", start_index )
                    end_index = len(line)
#                     print("end_index = ", end_index)
                    line_no += 1
                    nfeatures = line[11:15]
                    nfeatures = int(nfeatures)
#                     print("nfeatures = ",line[11:15])
        #         print("type of v(line) = ",type(line))
        #         print(line)
                else:
                    matching_list.append(line.rstrip('\n'))
    if (a == 2):
        with open(database_path + "matching2.txt") as f:
            line_no = 1
            # print("File opened")
            for line in f:
                if line_no == 1:
                    start_index = line.find(':')
#                     print("start_index = ", start_index )
                    end_index = len(line)
#                     print("end_index = ", end_index)
                    line_no += 1
                    nfeatures = line[11:15]
                    nfeatures = int(nfeatures)
#                     print("nfeatures = ",line[11:15])
        #         print("type of v(line) = ",type(line))
        #         print(line)
                else:
                    matching_list.append(line.rstrip('\n'))

    if (a == 3):
        with open(database_path + "matching3.txt") as f:
            line_no = 1
            # print("File opened")
            for line in f:
                if line_no == 1:
                    start_index = line.find(':')
#                     print("start_index = ", start_index )
                    end_index = len(line)
#                     print("end_index = ", end_index)
                    line_no += 1
                    nfeatures = line[11:15]
                    nfeatures = int(nfeatures)
#                     print("nfeatures = ",line[11:15])
        #         print("type of v(line) = ",type(line))
        #         print(line)
                else:
                    matching_list.append(line.rstrip('\n'))

    if (a == 4):
        with open(database_path + "matching4.txt") as f:
            line_no = 1
            print("File opened")
            for line in f:
                if line_no == 1:
                    start_index = line.find(':')
#                     print("start_index = ", start_index )
                    end_index = len(line)
#                     print("end_index = ", end_index)
                    line_no += 1
                    nfeatures = line[11:15]
                    nfeatures = int(nfeatures)
#                     print("nfeatures = ",line[11:15])
        #         print("type of v(line) = ",type(line))
        #         print(line)
                else:
                    matching_list.append(line.rstrip('\n'))

    if (a == 5):
        with open(database_path + "matching5.txt") as f:
            line_no = 1
            # print("File opened")
            for line in f:
                if line_no == 1:
                    start_index = line.find(':')
#                     print("start_index = ", start_index )
                    end_index = len(line)
#                     print("end_index = ", end_index)
                    line_no += 1
                    nfeatures = line[start_index + 2:end_index - 1]
                    nfeatures = int(nfeatures)
#                     print("nfeatures = ",line[11:15])
        #         print("type of v(line) = ",type(line))
        #         print(line)
                else:
                    matching_list.append(line.rstrip('\n'))
    final_list = []
    for i in range(0, len(matching_list)):
        current_row = matching_list[i]
        splitStr = current_row.split()
        current_row = []
        for j in splitStr:
            current_row.append(float(j))
        final_list.append(np.transpose(current_row))

    output_list = []

    for i in range(0, len(final_list)):
        new_row = []
        current_row = final_list[i]
        current_row = current_row[1:len(current_row)]
#         print("len_current_row",len(current_row))
#         print("early Current Row = ",current_row)
#         current_row = current_row
        res = np.where(current_row == b)
#         if(res[0])
#         print("res = ",res)
#         print(len(res[0]))
        if(len(res[0]) != 0):
            index = res[0][0]
#             print("index is = ",index)
            new_row.append(current_row[0])
            new_row.append(current_row[1])
            new_row.append(current_row[2])
            new_row.append(current_row[3])
            new_row.append(current_row[4])
            new_row.append(current_row[index + 1])
            new_row.append(current_row[index + 2])

        if (len(new_row) != 0):
            output_list.append(np.transpose(new_row))
#     print("final List", final_list)
    matching_list = output_list
    # print("Done extracting correspondance points. Format:R,G,B,x1,y1,x2,y2")
    # print("Number of correspondance points: ", len(matching_list))
    return matching_list
