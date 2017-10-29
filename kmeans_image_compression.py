import numpy as np
import matplotlib.pyplot as plt

import cv2
from sklearn.cluster import KMeans

def kmeans_image_compression(original_img, k):
    shape = original_img.shape[:3]                          # サイズ取得
    X = original_img.reshape(shape[0]*shape[1], shape[2])   # 行列から1列に変換する

    model = KMeans(n_clusters=k)                            # k-meansモデルをクラスタ数k個で作成
    model.fit(X)                                            # データXを解析
    centers = model.cluster_centers_.astype('uint8')        # クラスタ中心を得る(astypeでintにキャスト変換)
    Y = model.predict(X)                                    # クラスタリング結果Yを取得

    # 各ピクセルをクラスタ中心で置き換え
    X2 = np.copy(X)                                         # XのコピーX2を作成、X2を置き換えていく
    for i in range(Y.shape[0]):
        X2[i] = centers[Y[i]]                               # 各ピクセルをクラスタ中心で置き換え
    compressed_img = X2.reshape(shape)                      # 行列(画像)の形に戻す
    
    return compressed_img


if __name__ == '__main__':
    original_img = cv2.imread('./Balloon.jpg')
    
    # 各kに対して画像圧縮
    for k in [2,3,4,5,10,20,50,100]:
        compressed_img = kmeans_image_compression(original_img, k)                                          # 画像圧縮
        cv2.putText(compressed_img,'K='+str(k),(10,15),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1,cv2.LINE_AA)    # 文字(K=*)を左上に描画
        cv2.imwrite('./result'+str(k)+'.jpg', compressed_img)                                               # 画像保存

    cv2.putText(original_img,'ORIGINAL',(10,15),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1,cv2.LINE_AA)           # 元画像の左上に文字を描画
    cv2.imwrite('./result9999.jpg', original_img)                                                           # 文字を付け加えた元画像を保存

    #plt.imshow(cv2.cvtColor(compressed_img, cv2.COLOR_BGR2RGB))
    #plt.show()
