import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import *
import os
import tensorflow as tf
from keras import backend as K

cpu = tf.config.list_physical_devices("CPU")
tf.config.set_visible_devices(cpu)
print(tf.config.list_logical_devices())

# session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=32, inter_op_parallelism_threads=32)
# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# K.set_session(sess)
def getxy():
    targetKey = '货量'
    # 对场地进行编码

    df = pd.read_excel('附件1：物流网络历史货量数据.xlsx')
    df['场地1'] = df['场地1'].str.replace('DC', '')
    df['场地1'] = df['场地1'].astype('int64')
    df['场地2'] = df['场地2'].str.replace('DC', '')
    df['场地2'] = df['场地2'].astype('int64')

    df['日期'] = pd.to_datetime(df['日期'])
    df['日期'] = df['日期'] - df['日期'].min()
    df['日期'] = df['日期'].apply(lambda x: x.days)
    return df.drop(targetKey, axis=1), df[targetKey]
forceRetrain = True

modelPath = 'model1.h5'

X, Y = getxy()

model = None
shouldTrain = False
if not os.path.exists(modelPath) or forceRetrain:
    shouldTrain = True

# 创建一个保存模型的回调函数
checkpoint_path = "training-cp.ckpt"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=False,
                                                         verbose=1)

if not shouldTrain:
    print('加载模型')
    # 加载模型
    model = tf.keras.models.load_model(modelPath)
else:
    print('重新训练模型')

    # 定义模型
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dense(256 * 2, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    if os.path.exists(checkpoint_path):
        model = tf.keras.models.load_model(checkpoint_path)


    # 定义自定义指标函数
    def accuracy(y_true, y_pred):
        threshold = 0.6  # 指定阈值
        diff = tf.abs(y_true - y_pred)  # 计算预测值和真实值之差的绝对值
        return tf.reduce_mean(tf.cast(diff <= threshold, tf.float32))  # 统计正确分类的样本数占总样本数的比例


    # 编译模型
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics='mae'
                  )

    # 训练模型
    history = model.fit(X, Y, epochs=1000, validation_split=0.1, callbacks=[checkpoint_callback])

    # 绘制学习曲线
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b-', label='Training loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'b-', label='Training acc')
    plt.plot(epochs, val_acc, 'r-', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('Training and validation loss accuracy.svg')
    plt.show()

    # 可视化训练和测试误差
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    # plt.ylim([0, 10])
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig('Model loss.svg')
    plt.show()

    # 保存模型
    model.save(modelPath)

# 预测并可视化结果
y_pred = model.predict(X)
plt.scatter(Y, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.axis('square')
# plt.xlim([9e4, 1e5])
# plt.ylim([9e4, 1e5])
_ = plt.plot([-1e10, 1e10], [-1e10, 1e10])
if shouldTrain:
    plt.savefig('visualize result.svg')
plt.show()

print(f'r2_score: {r2_score(Y, y_pred)}')
print(f'mse: {mean_squared_error(Y, y_pred)}')
print(f'mae: {mean_absolute_error(Y, y_pred)}')
print(f'mape: {mean_absolute_percentage_error(Y, y_pred)}')
