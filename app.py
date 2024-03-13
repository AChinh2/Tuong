from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import tensorflow as tf



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', result=None)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']

     # Lưu file tải lên vào thư mục tạm thời
    file.save(file.filename)
    # Đọc thông tin của ảnh
    img = Image.open(file)
    #img = img.convert("RGB")
    img = img.convert("L")
    
    shape0 = img.size
    img = img.resize((180, 180))  # Thay đổi kích thước ở đây
    shape1 = img.size
    # Chuyển ảnh đã thay đổi kích thước thành mảng numpy
    img = np.array(img)
    img = img/255
    #shape2 = type(img)
    #img = np.expand_dims(img, axis=0)
    #img = np.expand_dims(img, axis=0)
    img = img.reshape(1,180,180,1)
    shape2 = img.shape
  
    # load model
    model = tf.keras.models.load_model(r"cnn.h5")
    
    # Xử lý file ở đây, ví dụ: đọc file ảnh và thực hiện xử lý
    # Đây chỉ là một ví dụ đơn giản
    
    label = ["NORMAL", "PNEUMONIA"]
    y_pred = model.predict(img)
    if y_pred > 0.8:
        y_pred = 1
    else:
        y_pred = 0
    result = label[y_pred]
    
    #result = 0
    return render_template('index.html', y_pred=y_pred, result=result)
if __name__ == '__main__':
    app.run(debug=True)
