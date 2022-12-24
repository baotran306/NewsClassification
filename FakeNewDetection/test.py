import tkinter as tk
from tkinter import ttk

# Creating tkinter window and set dimensions
window = tk.Tk()
window.title('Combobox')
window.geometry('500x250')


def callbackFunc(event):
    country = event.widget.get()
    print(country)


# label text for title
ttk.Label(window, text="Choose the country and vote for them",
          background='cyan', foreground="black",
          font=("Times New Roman", 15)).grid(row=0, column=1)

# Set label
ttk.Label(window, text="Select the Country :",
          font=("Times New Roman", 12)).grid(column=0,
                                             row=5, padx=5, pady=25)

# Create Combobox
n = tk.StringVar()
country = ttk.Combobox(window, width=27, textvariable=n)

# Adding combobox drop down list
country['values'] = (' India',
                     ' China',
                     ' Australia',
                     ' Nigeria',
                     ' Malaysia',
                     ' Italy',
                     ' Turkey',
                     ' Canada')

country.grid(column=1, row=5)
country.current()
country.bind("<<ComboboxSelected>>", callbackFunc)

window.mainloop()
