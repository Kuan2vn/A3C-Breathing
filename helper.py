import matplotlib.pyplot as plt
from IPython import display
import shutil
import os

plt.ion()

def plot_as(scores, loss): # mean_scores1)

    # display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()

    plt.subplot(2, 1, 1)
    plt.title('Training Reward...')
    plt.xlabel('Number of Games')
    plt.ylabel('Reward')
    plt.plot(scores, label = 'reward')
    

    # plt.ylim(ymin=-250, ymax=600)
    # plt.legend()
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))

    plt.subplot(2, 1, 2)
    plt.title('Training Loss...')
    plt.xlabel('Number of Games')
    plt.ylabel('Loss')

    plt.plot(loss, label = 'loss')
    plt.text(len(loss)-1, loss[-1], str(loss[-1]))

    plt.tight_layout()

    plt.show(block=False)
    plt.pause(.1)

def copy_file_with_number(source_file, destination_folder, new_number):
    # Kiểm tra xem tệp tin gốc tồn tại hay không
    if os.path.exists(source_file):
        # Kiểm tra xem thư mục đích đã tồn tại chưa, nếu chưa thì tạo mới
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        
        # Tách tên tệp và phần mở rộng
        file_name, file_extension = os.path.splitext(os.path.basename(source_file))
        
        # Thêm số vào tên tệp
        new_file_name = f"{file_name}_{new_number}{file_extension}"
        
        # Đường dẫn đầy đủ đến tệp mới
        destination_file = os.path.join(destination_folder, new_file_name)
        
        # Sao chép tệp tin
        shutil.copy(source_file, destination_file)
        return destination_file
    else:
        return None