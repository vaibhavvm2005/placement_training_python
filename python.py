ipc_data = {
    '300': 'murder',
    '500': 'theft',
    '700': 'rape',
    '100': 'fraud',
    '200': 'arson',
    '400': 'assault',
    '600': 'kidnapping',
    '800': 'robbery',
    '900': 'burglary',
    '1000': 'homicide'
}

print("Welcome to my ipc_data")

while True:
    sec = input("Enter the section or type exit: ")

    if sec.lower() == "exit":
        print("Exiting...")
        break

    if sec in ipc_data:
        print("Section", sec + ":", ipc_data[sec])
    else:
        print("Section not found")