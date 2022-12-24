import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import process_dataframe
import NaiveBayes
import kNN
from sklearn.metrics import accuracy_score

x = 1
blue_color = '#158aff'
gray_color = '#c3c3c3'
red_color = '#FF3333'
height_frame_df = 230
width_frame_df = 785
nb_algorithm_name = 'Naive Bayes'
knn_algorithm_name = 'k-Nearest Neighbor'
myNB = NaiveBayes.MyGaussianNB()
myKnn = kNN.my_kNN()
df_train = pd.DataFrame()
df_test = pd.DataFrame()
df_valid = pd.DataFrame()
ans_fake_df = pd.DataFrame()
ans_real_df = pd.DataFrame()
current_algorithm = ''
root = tk.Tk()
root.geometry('1000x1000')
root.title('Ứng dụng phân loại tin tức')
root.resizable(0, 0)


def file_dialog(label_file):
    filename = filedialog.askopenfilename(initialdir='/',
                                          title="Chon file",
                                          filetype=(("csv files", "*.csv"), ("All Files", "*.*")))
    label_file['text'] = filename


def get_algorithm_name(event):
    global current_algorithm
    current_algorithm = event.widget.get()


def show_combobox_algorithm(frame):
    cbb_algorithm = ttk.Combobox(frame, state='readonly')
    cbb_algorithm['values'] = (nb_algorithm_name, knn_algorithm_name)
    cbb_algorithm.current(0)
    # cbb_algorithm.place(x=30, y=650, width=140, height=20)
    cbb_algorithm.place(rely=0.10, relx=0.80)
    cbb_algorithm.bind("<<ComboboxSelected>>", get_algorithm_name)
    return cbb_algorithm


def start_train(label_file_accuracy):
    # print('start train')
    print(current_algorithm)
    # print(df_train.head())
    # print(df_valid.head())
    if not df_train.empty:
        # global myNB, myKnn
        x_train_input, y_label_train = process_dataframe.process_df(df_train, type='train')
        if current_algorithm == knn_algorithm_name:
            myKnn.fit(x_train_input, y_label_train)
        else:
            myNB.fit(x_train_input, y_label_train)
        # if have valid dataset, predict and calculate accuracy in valid dataset
        if not df_valid.empty:
            x_valid_input, y_label_valid = process_dataframe.process_df(df_valid)
            if current_algorithm == knn_algorithm_name:
                valid_predict = myKnn.predict(x_valid_input)
            else:
                valid_predict = myNB.predict(x_valid_input)
            score = accuracy_score(y_label_valid, valid_predict)
        # Show infor success
            label_file_accuracy['text'] = f'Độ chính xác trên tập validation theo {current_algorithm} là:' \
                                          f' {round(score, 2)}'
        tk.messagebox.showinfo("Information", f"Hoàn tất huấn luyện bằng thuật toán {current_algorithm},"
                               f"có thể tiến hành dự đoán.")
    else:
        tk.messagebox.showerror("Information", "Chưa load đủ file")


def start_predict(view_show_real, view_show_fake):
    print(current_algorithm)
    if not df_test.empty:
        # global myNB, myKnn
        signal = 0
        valid_predict = []
        df_origin_test = df_test.copy()
        x_test_input = process_dataframe.process_df(df_test, type='test')
        try:
            if current_algorithm == knn_algorithm_name:
                valid_predict = myKnn.predict(x_test_input)
            else:
                valid_predict = myNB.predict(x_test_input)
        except:
            signal = 1
            tk.messagebox.showerror("Information", f"Bạn chưa huấn luyện mô hình bằng thuật toán {current_algorithm},"
                                    f" vui lòng huấn luyện rồi thử lại")
        print(valid_predict)
        print(type(valid_predict))
        df_label = pd.DataFrame(valid_predict, columns=['label'], index=df_origin_test.index)
        full_df = pd.concat([df_origin_test, df_label], axis=1)
        print(full_df)
        show_data_tree_view(view_show_fake, full_df.loc[full_df['label'] == 0])
        show_data_tree_view(view_show_real, full_df.loc[full_df['label'] == 1])
        if signal == 0:
            tk.messagebox.showinfo("Information", f"Hoàn tất dự đoán bằng thuật toán {current_algorithm}")
    else:
        tk.messagebox.showerror("Information", "Chưa load đủ file")


def load_data(tree_view, label_file, status):
    global df_train, df_valid, df_test
    file_path = label_file['text']
    try:
        data_file = f"{file_path}"
        df_csv = pd.read_csv(data_file)
    except ValueError:
        tk.messagebox.showerror("Information", "File chọn không hợp lệ")
        return None
    except FileNotFoundError:
        tk.messagebox.showerror("Information", f"Không tìm thấy {file_path}")
        return None

    show_data_tree_view(tree_view, df_csv)
    if status == 1:
        df_train = get_data_tree_view(tree_view)
        print(11)
    elif status == 2:
        df_valid = get_data_tree_view(tree_view)
        print(22)
    else:
        df_test = get_data_tree_view(tree_view, type_use='test')
        print(33)
    return None


def clear_data(tree_view):
    tree_view.delete(*tree_view.get_children())


def show_data_tree_view(tree_view_show, df_data):
    clear_data(tree_view_show)
    tree_view_show["column"] = list(df_data.columns)
    tree_view_show["show"] = "headings"
    for col in tree_view_show["column"]:
        tree_view_show.heading(col, text=col)
    df_rows = df_data.to_numpy().tolist()
    for row in df_rows:
        tree_view_show.insert("", "end", values=row)


def get_data_tree_view(treeview, type_use='train'):
    try:
        row_list = []
        if type_use == 'train':
            columns = ["title", "text", "subject", "date", 'label']
        else:
            columns = ["title", "text", "subject", "date"]
        for row in treeview.get_children():
            row_list.append(treeview.item(row)["values"])
        treeview_df = pd.DataFrame(row_list, columns=columns)
        return treeview_df
    except:
        tk.messagebox.showerror("Information", "Các cột trong file không hợp lệ, vui lòng đọc lại hướng dẫn")
        return None


def hide_indicate(lb):
    lb.config(bg=gray_color)


def load_tree_view(frame):
    # Load tree view
    tree_view = ttk.Treeview(frame)
    tree_view.place(relheight=1, relwidth=1)

    tree_scroll_x = tk.Scrollbar(frame, orient="horizontal", command=tree_view.xview)
    tree_scroll_y = tk.Scrollbar(frame, orient="vertical", command=tree_view.yview)
    tree_view.configure(xscrollcommand=tree_scroll_x.set, yscrollcommand=tree_scroll_y.set)
    tree_scroll_x.pack(side=tk.BOTTOM, fill="x")
    tree_scroll_y.pack(side=tk.RIGHT, fill="y")
    return tree_view


def config_frame_show(frame):
    frame_show = tk.Frame(frame)
    frame_show.place(x=0, y=0)
    frame_show.pack(side=tk.TOP)
    frame_show.pack_propagate(False)
    frame_show.configure(height=800, width=800)
    return frame_show


def choose_file_label_view(frame, tree_view, status_frame, text_label, text_bt_choose='Browser file',
                           text_bt_load='Load file',
                           text_lb_file='No file selected', rely_frame=0.62):
    file_frame = tk.LabelFrame(frame, text=text_label,
                               highlightbackground='black', highlightthickness=2)
    file_frame.place(height=60, width=785, rely=rely_frame, relx=0.005)

    label_file = ttk.Label(file_frame, text=text_lb_file)
    label_file.place(rely=0, relx=0)

    button_choose = tk.Button(file_frame, text=text_bt_choose, command=lambda: file_dialog(label_file))
    button_choose.place(rely=0.02, relx=0.7)

    button_load = tk.Button(file_frame, text=text_bt_load, command=lambda: load_data(tree_view, label_file, status_frame))
    button_load.place(rely=0.02, relx=0.9)


def process_train(frame):
    global current_algorithm
    file_frame = tk.LabelFrame(frame, text="",
                               highlightbackground='black', highlightthickness=2)
    file_frame.place(height=100, width=785, rely=0.82, relx=0.005)

    label_file = ttk.Label(file_frame, text="", foreground=red_color)
    label_file.place(rely=0.35, relx=0)

    # Show label and combobox algrorithm
    label_algorithm = ttk.Label(file_frame, text='Chọn thuật toán: ')
    label_algorithm.place(rely=0.10, relx=0.67)
    combobox_algorithm = show_combobox_algorithm(file_frame)
    current_algorithm = combobox_algorithm.get()

    button_train = tk.Button(file_frame, text="Huấn luyện", font=('Bold', 12), fg=blue_color,
                             command=lambda: start_train(label_file))
    button_train.place(rely=0.45, relx=0.8)


def process_test(frame, view_real_show, view_fake_show):
    global current_algorithm
    file_frame = tk.LabelFrame(frame, text="",
                               highlightbackground='black', highlightthickness=2)
    file_frame.place(height=80, width=785, rely=0.82, relx=0.005)

    label_file = ttk.Label(file_frame, text="", foreground=red_color)
    label_file.place(rely=0.35, relx=0)
    print(df_test.head())
    button_train = tk.Button(file_frame, text="Dự đoán", font=('Bold', 12), fg=blue_color,
                             command=lambda: start_predict(view_show_real=view_real_show, view_show_fake=view_fake_show))
    button_train.place(rely=0.35, relx=0.8)


def show_indicate(lb, show_page, use_algorithm):
    hide_indicate(train_indicate)
    hide_indicate(test_indicate)
    hide_indicate(instruction_indicate)
    lb.config(bg=blue_color)
    delete_pages()
    print(use_algorithm)
    show_page()


def train_page():
    global df_train
    global df_valid

    # train_frame = tk.Frame(main_frame)
    # train_frame.place(x=0, y=0)
    # train_frame.pack(side=tk.TOP)
    # train_frame.pack_propagate(False)
    # train_frame.configure(height=800, width=800)
    # lb = tk.Label(train_frame, text='Phần huấn luyện', font=('Bold', 15))
    # lb.pack()
    train_frame = config_frame_show(main_frame)

    show_train_frame = tk.LabelFrame(train_frame, text="Tập dữ liệu huấn luyện",
                                     highlightbackground='black', highlightthickness=2)
    show_train_frame.place(height=height_frame_df, width=width_frame_df, rely=0.005, relx=0.005)
    view_train = load_tree_view(show_train_frame)

    show_valid_frame = tk.LabelFrame(train_frame, text="Tập dữ liệu kiểm thử",
                                     highlightbackground='black', highlightthickness=2)
    view_valid = load_tree_view(show_valid_frame)
    show_valid_frame.place(height=height_frame_df, width=width_frame_df, rely=0.32, relx=0.005)

    if not df_train.empty or not df_valid.empty:
        show_data_tree_view(view_train, df_train)
        show_data_tree_view(view_valid, df_valid)

    # file_frame_train = tk.LabelFrame(train_frame, text='Chọn file huấn luyện',
    #                                  highlightbackground='black', highlightthickness=2)
    # file_frame_train.place(height=60, width=785, rely=0.7, relx=0.005)
    #
    # button_choose_train = tk.Button(file_frame_train, text='Browser file')
    # button_choose_train.place(rely=0.02, relx=0.7)
    #
    # button_load_train = tk.Button(file_frame_train, text='Load file')
    # button_load_train.place(rely=0.02, relx=0.9)
    #
    # label_file_train = ttk.Label(file_frame_train, text='No file selected')
    # label_file_train.place(rely=0, relx=0)

    choose_file_label_view(train_frame, view_train, status_frame=1, text_label='Chọn file huấn luyện')

    choose_file_label_view(train_frame, view_valid, status_frame=2, text_label='Chọn file kiểm thử(Optional)',
                           rely_frame=0.72)

    # file_frame_valid = tk.LabelFrame(train_frame, text='Chọn file kiểm thử',
    #                                  highlightbackground='black', highlightthickness=2)
    # file_frame_valid.place(height=60, width=785, rely=0.8, relx=0.005)
    #
    # button_choose_valid = tk.Button(file_frame_valid, text='Browser file')
    # button_choose_valid.place(rely=0.02, relx=0.7)
    #
    # button_load_valid = tk.Button(file_frame_valid, text='Load file')
    # button_load_valid.place(rely=0.02, relx=0.9)
    #
    # label_file_valid = ttk.Label(file_frame_valid, text='No file selected')
    # label_file_valid.place(rely=0, relx=0)
    process_train(train_frame)

    train_frame.pack(pady=20)


def test_page():
    test_frame = config_frame_show(main_frame)

    show_test_frame = tk.LabelFrame(test_frame, text="Tập dữ liệu dự đoán",
                                    highlightbackground='black', highlightthickness=2)
    show_test_frame.place(height=180, width=width_frame_df, rely=0.005, relx=0.005)
    view_test = load_tree_view(show_test_frame)

    choose_file_label_view(test_frame, view_test, status_frame=3, text_label='Chọn file dự đoán', rely_frame=0.23)

    label_file = ttk.Label(test_frame, text='KẾT QUẢ', font=('Bold', 15), foreground=blue_color)
    label_file.place(rely=0.31, relx=0.005)

    show_real_frame = tk.LabelFrame(test_frame, text="Tập tin tức thật",
                                    highlightbackground='black', highlightthickness=2)
    show_real_frame.place(height=180, width=width_frame_df, rely=0.35, relx=0.005)
    view_real_df = load_tree_view(show_real_frame)

    show_fake_frame = tk.LabelFrame(test_frame, text="Tập tin tức giả",
                                    highlightbackground='black', highlightthickness=2)
    show_fake_frame.place(height=180, width=width_frame_df, rely=0.58, relx=0.005)
    view_fake_df = load_tree_view(show_fake_frame)
    process_test(test_frame, view_real_show=view_real_df, view_fake_show=view_fake_df)

    test_frame.pack(pady=20)


def instruction_page():
    instruction_frame = config_frame_show(main_frame)
    lb_infor_main = tk.Label(instruction_frame, text='Phần hướng dẫn', font=('Bold', 15))
    lb_infor_main.pack()

    lb_infor_sub = tk.Label(instruction_frame, text='Ứng dụng gồm có 2 phần chính là: HUẤN LUYỆN và DỰ ĐOÁN',
                            font=('Bold', 13))
    lb_infor_sub.pack()
    description =\
    """
       - Huấn luyện:
       \t+ Có màn hình hiển thị tập dữ liệu huấn luyện và tập dự liệu kiểm thử.
       \t+ Tập dữ liệu huấn luyện bắt buộc phải có để mô hình có thể huấn luyện được.
       \t+ Tập dữ liệu kiểm thử tùy chọn, có thể sử dụng để kiểm tra độ chính xác mô hình trên tập này.
       \t+ Để load tập dữ liệu vào, click chọn "browser" để chọn file, sau đó chọn "load file".
       \t+ Chú ý: File được chọn phải là file .csv, gồm có các cột "title", "text", "subject", "date", "label".
       \t+ Mô hình đang sử dụng hai thuật toán khác nhau là KNN và Naive Bayes.
       \t+ Tùy chọn một trong hai thuật toán, sau đó chọn nút "Huấn luyện" để bắt đầu quá trình huấn luyện.
       \t+ Nếu có sử dụng dữ liệu kiểm thử, ứng dụng sẽ trả ra độ chính xác trên tập này sau quá trình huấn luyện.
       
       - Dự đoán:
       \t+ Tương tự quá trình trên, chọn file dữ liệu cần dự đoán để mô hình tiến hành dự đoán.
       \t+ File được chọn phải là file .csv, gồm có các cột "title", "text", "subject", "date".
       \t+ Sau khi dự đoán thành công, tập tin thức thật sẽ hiện trên view tập tin tức thật và ngược lại.
    """
    lb = tk.Label(instruction_frame, text=description, font=('Bold', 11), justify='left')
    lb.place(relx=0.005, rely=0.1)
    instruction_frame.pack(pady=5)


def delete_pages():
    for frame in main_frame.winfo_children():
        frame.destroy()


if __name__ == '__main__':
    option_frame = tk.Frame(root, bg='#c3c3c3')

    # combobox_algorithm = show_combobox_algorithm(option_frame)
    # current_algorithm = combobox_algorithm.get()
    train_button = tk.Button(option_frame, text='HUẤN LUYỆN', font=('Bold', 15), fg=blue_color, bd=0,
                             command=lambda: show_indicate(train_indicate, train_page, current_algorithm))
    train_button.place(x=30, y=50, width=140, height=40)
    train_indicate = tk.Label(option_frame, text='', bg=gray_color)
    train_indicate.place(x=3, y=50, width=5, height=40)

    test_button = tk.Button(option_frame, text='DỰ ĐOÁN', font=('Bold', 15), fg=blue_color, bd=0,
                            command=lambda: show_indicate(test_indicate, test_page, current_algorithm))
    test_button.place(x=30, y=250, width=140, height=40)
    test_indicate = tk.Label(option_frame, text='', bg=gray_color)
    test_indicate.place(x=3, y=250, width=5, height=40)

    instruction_button = tk.Button(option_frame, text='HƯỚNG DẪN', font=('Bold', 15), fg=blue_color, bd=0,
                                   command=lambda: show_indicate(instruction_indicate, instruction_page, current_algorithm))
    instruction_button.place(x=30, y=450, width=140, height=40)
    instruction_indicate = tk.Label(option_frame, text='', bg=gray_color)
    instruction_indicate.place(x=3, y=450, width=5, height=40)

    option_frame.pack(side=tk.LEFT)
    option_frame.pack_propagate(False)
    option_frame.configure(width=200, height=1000)

    main_frame = tk.Frame(root, highlightbackground='black', highlightthickness=2)
    main_frame.pack(side=tk.LEFT)
    main_frame.pack_propagate(False)
    main_frame.configure(height=1000, width=800)

    root.mainloop()
