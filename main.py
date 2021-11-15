import model
from model import StandardScaler

#'Oral Level', 'Written Level', 'Listening', 'Speaking', 'Reading', 'Writing' , 'Current GPA', '18-19 Overall ELPAC'

def inputParameters():
    global xNew
    oral = input("Enter the student's Oral Level: ")
    written = input("Enter the student's Written Level: ")
    listening = input("Enter the student's Listening Level: ")
    speaking = input("Enter the student's Speaking Level: ")
    reading = input("Enter the student's Reading Level: ")
    writing = input("Enter the student's Writing Level: ")
    current_gpa = input("Enter the student's Current GPA:  ")
    elpac = input("Enter the student's 18-19 Elpac: ")
    xNew = [[int(oral), int(written), int(listening), int(speaking), int(reading), int(writing), float(current_gpa), int(elpac)]]
    return print("Inputting student's attributes completed")
    
a = model.func()
inputParameters()
input = xNew
print(a.predict(input))