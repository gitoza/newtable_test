import numpy as np
import os
#import decimal
import math
import requests

import datetime as dt

class Grib_Decode:
    def __init__(self, file_path=None, data=None):
        self.file_path = file_path
        self.data = data

        if file_path:
            self.load_grib2_from_file()
        elif data is not None:
            self.load_grib2_from_memory(data)
        else:
            raise ValueError("file_pathまたはdataを設定してください")
        
        self.get_start_index()
        self.get_params()
        self.initialize_lat_lon()

    def load_grib2_from_file(self):
        #gribデータ読み込み
    #    print("file_load")
        with open(self.file_path, 'rb') as fp:
            fp = open(self.file_path, 'rb')
            file_size = os.stat(self.file_path).st_size

        self.ndarr1 = np.fromfile(fp, np.uint8, file_size)

    def load_grib2_from_memory(self, data):
        print("memory_load")
        self.ndarr1 = np.frombuffer(data, dtype=np.uint8)

    # byte配列を整数に変換する
    def array_to_int(self,array):
        hex = ''.join(format(a, '02x') for a in array)
        return int(hex, 16)

    # 同じ値が連続するサイズ（ランレングス）を取得する
    # input: numpy array
    # V    : max value of the level
    def get_runlength(self, input, V):
        N = 8
        L = (2**N) - 1 - V
        indices = np.arange(input.size)  # 各要素のインデックス配列を生成
        terms = (L ** indices) * (input - (V + 1))  
        sum_result = 1 + np.sum(terms)
        return int(sum_result)


    # 各節の開始インデックスを配列にして取得する
    def get_start_index(self):
        index = np.zeros(8, dtype='int64')

        index[0] = 16
        index[1] = 21 + int(index[0])
        index[2] =  0 + int(index[1])
        index[3] = 72 + int(index[2])
        index[4] = self.array_to_int(self.ndarr1[index[3]:index[3]+4]) + index[3]
        index[5] = self.array_to_int(self.ndarr1[index[4]:index[4]+4]) + index[4]
        index[6] =  6 + index[5]
        index[7] = self.array_to_int(self.ndarr1[index[6]:index[6]+4]) + index[6]
        
        self.index = index
        
        return self.index

    # 各種パラメータをデコードする
    def get_params(self):
        ndarr1 =self.ndarr1
        index = self.index
        #index = self.index
        self.V = self.array_to_int(ndarr1[index[4]+12:index[4]+14])
        self.M = self.array_to_int(ndarr1[index[4]+14:index[4]+16])
        self.num_lat = self.array_to_int(ndarr1[index[2]+30:index[2]+34])
        self.num_lon = self.array_to_int(ndarr1[index[2]+34:index[2]+38])

        self.first_lat = self.array_to_int(ndarr1[index[2]+46:index[2]+50])/10**6
        self.first_lon = self.array_to_int(ndarr1[index[2]+50:index[2]+54])/10**6
        self.last_lat = self.array_to_int(ndarr1[index[2]+55:index[2]+59])/10**6
        self.last_lon = self.array_to_int(ndarr1[index[2]+59:index[2]+63])/10**6

        self.increment_lon = self.array_to_int(ndarr1[index[2]+63:index[2]+67])/10**6
        self.increment_lat = self.array_to_int(ndarr1[index[2]+67:index[2]+71])/10**6
        
        #print(f"{self.increment_lon},{self.increment_lat}")
        
        #データ代表値の尺度因子(位置207番目)
        self.factor = int(ndarr1[index[4]+16])

    #読み込んだデータと同じサイズの配列に緯度経度
    def initialize_lat_lon(self):
        # dataの形状に基づいて緯度経度配列を初期化
        self.lats = np.zeros((self.num_lon, self.num_lat))
        self.lons = np.zeros((self.num_lon, self.num_lat))

        # 各格子点の緯度経度を取得
        for i in range(self.num_lat):
            for j in range(self.num_lon):
                self.lats[j, i], self.lons[j, i] = self.get_lat_lon(j, i)


    # レベル値配列を取得する
    def get_level_array(self):
        index = self.get_start_index()
        levarr=np.copy(self.ndarr1[index[4]+17:index[5]]) # 208:208+int(sec5-sec4)])
        
        Level = np.zeros(self.M) # M = np.size(levarr)/2
        
        j=0
        for i in range(0, np.size(levarr), 2):
            if i == np.size(levarr) -1:
                break
            Level[j] = self.array_to_int(levarr[i:i+2])/10**self.factor
            j += 1

        missing = 0.0 #欠測値
        self.Level = np.insert(Level, 0, missing)

    def decode_grib2(self,data, V, Level,num_lat,num_lon):
        V, Level , num_lat ,num_lon = self.V , self.Level , self.num_lat , self.num_lon
        rain_level = 0
        runlen_n = out_index = R = 0

        runlen_r = np.zeros(np.size(Level)-1)
        out_data = np.zeros(num_lat * num_lon) #格子数 2560*3360

        for i in range(np.size(data)):

            if out_index >= np.size(out_data) or i == (np.size(data)):
                break

            if data[i] > V:
                runlen_r[runlen_n] = int(data[i])
                runlen_n += 1
                rain_level = int(data[i-runlen_n])

            elif data[i] <= V:
                runlen_n = 0
                index = np.copy(runlen_r>0)#0以外の配列を取り出す
                R = self.get_runlength(runlen_r[index], V)

                for k in range(R):
                    out_data[out_index] = Level[rain_level]

                    out_index += 1
                
                runlen_r = np.zeros(np.size(Level))
                rain_level = int(data[i])

        out_data = np.reshape(out_data, (num_lon, num_lat))
        return out_data

    def read_grib2(self):
        self.get_params()
        self.get_level_array()
        #ndarr1 = self.ndarr1
        #index = self.index
        #V, M, factor, _ ,_ = self.get_params(ndarr1, index)
        #Level = self.get_level_array(ndarr1, index, M, factor)
        
        data = np.copy(self.ndarr1[self.index[6]+4:self.index[7]])
        result = self.decode_grib2(data, self.V, self.Level,self.num_lat,self.num_lon)

        return result
    
    def get_lat_lon(self,lat_index,lon_index):
        const_lat_start = self.first_lat
        const_lat_increment = self.increment_lat
        const_lon_start = self.first_lon
        const_lon_increment = self.increment_lon

        self.lat = const_lat_start - lat_index * const_lat_increment
        self.lon = const_lon_start + lon_index * const_lon_increment
        
        return self.lat, self.lon


    #緯度経度からindex番号を計算する関数
    def get_index(self,lat, lon):
        const_lat_start = self.first_lat
        const_lat_increment = self.increment_lat
        const_lon_start = self.first_lon
        const_lon_increment = self.increment_lon

        # 逆計算
        lat_index = round((const_lat_start - lat) / const_lat_increment)
        lon_index = round((lon - const_lon_start) / const_lon_increment)+1
       # print(lat_index,lon_index)
        return (lat_index, lon_index)



"""

def grib2csv(data, output_file):
    #out_data = Grib_Decode.read_grib2(input_file)

    #print('解析雨量：鷹巣=',out_data[934][1791], sep='', end='')
    #print('秋田=',out_data[995][1769], sep='', end='')
    #print('横手=',out_data[1042][1862]) 

    #結果出力
    np.set_printoptions(precision=5, suppress=True) #有効桁３桁で丸める 指数表示禁止
    np.savetxt(output_file, data, delimiter=',' , fmt='%f')

# yokote
#  get_lat_lon((1042,1862)) => (39.3514, 141.275)
def get_lat_lon(index):
    const_lat_st = 48
    const_lat_tick = 0.0083
    const_lon_st = 118
    const_lon_tick = 0.0125

    lat = const_lat_st - index[0] * const_lat_tick
    lon = const_lon_st + index[1] * const_lon_tick
    
    return (lat, lon)

#緯度経度からindex番号を計算する関数
def get_index(lat, lon):
    const_lat_st = 48
    const_lat_tick = 0.05
    const_lon_st = 118
    const_lon_tick = 0.0625

    # 逆計算
    lat_index = round((const_lat_st - lat) / const_lat_tick)
    lon_index = round((lon - const_lon_st) / const_lon_tick)+1
    print(lat_index,lon_index)
    return (lat_index, lon_index)


def get_max_lat_lon(index, area):
    lat_lon = []    

    for each_index in index:
        each_lat, each_lon = get_lat_lon(each_index)
        if area[0] <= each_lat and each_lat <= area[1] and area[2] <= each_lon and each_lon <= area[3]:
            lat_lon.append((each_lat, each_lon))

    return lat_lon

#Z__C_RJTD_20220202000000_SRF_GPV_Gll5km_Psdlv_ANAL_grib2.bin
def gen_grib_url(year,  month,  day,hour , minute):

"""

"""int型の日時を引数  gribファイルが保存されているURLを生成する関数
       取得対象によって要変更

    Args:
        target_time (_type_):datetime型 

    Returns:
        _type_: str型 URLが生成される
"""
"""
    target_time = dt.datetime(year,  month,  day,hour , minute)

    #http://lunar1.fcd.naps.kishou.go.jp/srf/Grib2/Rtn/swi10/2023/01/01/Z__C_RJTD_20230101000000_SRF_GPV_Ggis1km_Psw_Aper10min_ANAL_grib2.bin

    header_url = "http://lunar1.fcd.naps.kishou.go.jp/srf/Grib2/Rtn/swi10/"
    tail_url = "00_SRF_GPV_Ggis1km_Psw_Aper10min_ANAL_grib2.bin"

    formatted_string = target_time.strftime("%Y%m%d%H%M")
    dir_name = target_time.strftime("%Y/%m/%d")

    file_name = header_url + dir_name +"/Z__C_RJTD_" +formatted_string +tail_url
    
    return file_name

def download_grib2(url):
    # プロキシサーバーの情報（プロキシは取得したいデータに合わせて都度変える）
    proxy = {
        'http': '172.27.232.11:3128',  # HTTPプロキシ
        'https': '172.27.232.11:3128'  # HTTPSプロキシ
    }

    # ダウンロードとメモリ上での処理
    response = requests.get(url, proxies=proxy).content

    return response

if __name__ == "__main__": 
    url = gen_grib_url(2022, 1, 1, 0 ,0 )
    #DL_data = download_grib2(url)


    path = "Z__C_RJTD_20240101000000_SRF_GPV_Ggis1km_Psw_Aper10min_ANAL_grib2.bin"

    data = Grib_Decode(path).read_grib2()
    #data = read_grib2(path)|
    #del DL_data
    
"""

#開始時間(UTC)==================================
""" 
    start_time = dt.datetime(2020,  9,  1, 16, 0)
    end_time   = dt.datetime(2020,  9, 10, 17, 0)
    time_step  = dt.timedelta(minutes=30)
    
    area_akita    = (38.8, 40.5, 139.6, 141.0)
    area_yamagata = (37.7, 39.2, 139.5, 140.7)

    #File Name
    root_dir = '/mnt/nas6/Grib2/ra/'

    #Mask
    mask_akita    = get_mask(area_akita)
    mask_yamagata = get_mask(area_yamagata)

    target_time = start_time
    output_akita = [['datetime', 'max', 'lat', 'lon']]
    output_yamagata = [['datetime', 'max', 'lat', 'lon']]

    while target_time <= end_time:
        print(target_time)
        input_name = root_dir + target_time.strftime('%Y/%m/%d/Z__C_RJTD_%Y%m%d%H%M00_SRF_GPV_Ggis1km_Prr60lv_ANAL_grib2.bin')
        r1 = read_grib2(input_name)

        #akita
        r1_max       = np.max(r1[mask_akita])
        r1_max_index = list(zip(*np.where(r1 == r1_max)))
        r1_max_pos   = get_max_lat_lon(r1_max_index, area_akita)
        output_akita.append([target_time, r1_max, r1_max_pos])

        #yamagata
        r1_max       = np.max(r1[mask_yamagata])
        r1_max_index = list(zip(*np.where(r1 == r1_max)))
        r1_max_pos   = get_max_lat_lon(r1_max_index, area_yamagata)
        output_yamagata.append([target_time, r1_max, r1_max_pos])

        target_time = target_time + time_step

    output_dir = 'csv/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_name = start_time.strftime('%Y%m%d%H%M') + '-' + end_time.strftime('%Y%m%d%H%M.csv')

    #grib2csv(input_name, output_dir + output_name)
    #結果出力
    #np.set_printoptions(precision=5, suppress=True) #有効桁３桁で丸める 指数表示禁止
    #np.savetxt('akita' + output_name, output_akita, delimiter=',' , fmt='%f')
    #np.savetxt('yamagata' + output_name, output_yamagata, delimiter=',' , fmt='%f')
    with open(output_dir + 'akita' + output_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(output_akita)

    with open(output_dir + 'yamagata' + output_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(output_yamagata)

 """
