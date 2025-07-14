# Laporan Proyek Machine Learning Analisis Prediktif Energy Consumption - Mohamad Baskoro Aji

## Domain Proyek

Penggunaan Energi sangat krusial pada era sekarang untuk seperti penggunaan rumah tangga. Banyaknya permintaan dari rumah tangga dipengaruhi oleh banyak faktor seperti Lokasi, Cuaca, Karakteristik Rumah, dan Sosio-Ekonomi dan waktu dataset ini dibuat yaitu pada tahun 2022 banyak orang orang masih bekerja dari rumah.Oleh karena itu dibutuhkannya analisis prediktif untuk mengukur seberapa besar penggunaan energi sehari hari agar pemerintah atau perusahaan penyedia layanan listrik/energi dapat memprediksi pola pada penggunaan energi, efisiensi dalam operasional, dan mengurangi dampak buruk ke lingkungan

  
  Referensi:
  -- 
 - [Energy Consumption Analysis and Characterization of the Residential Sector in the US towards Sustainable Development](https://www.mdpi.com/1996-1073/17/11/2789)
 - [An analysis of monthly household energy consumption among single-family residences in Texas](https://www.sciencedirect.com/science/article/abs/pii/S0301421513012536)


## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- **Masalah 1**: Bagaimana Memprediksi penggunaan energi berdasarkan beberapa kondisi seperti temperatur, HVAC(Heating, Ventilation, and Air Conditioning), Karakteristik Bangunan, dan berapa banyak orang yang menempatinya?
- **Masalah 2**: Apa saja faktor terbesar dalam tingginya penggunaan energi?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Membangun sebuah model regresi machine learning berdasarkan faktor faktor yang ada untuk mendukung keputusan
- Melakukan analisis dan identifikasi terhadap faktor yang paling berpengaruh dalam penggunaan energi


    ### Solution statements
    - **Solution 1** : Menggunakan 5 algoritma berbeda dimana di setiap algoritma akan di ukur berdasarkan metrik penilaian:
    - SVR (menangkap hubungan non-linier)
    - RandomForest (Menangkap hubungan non-linear)
    - Gradient Boosting (Optimasi bertahap)
    - XGBoost (Regularisasi untuk prevent overfitting)
    - LightGBM (Efisiensi komputasi)
    - **Solution 2**: Menggunakan Feature Importance pada setiap algoritma untuk menangkap faktor faktor besar yang mempengaruhi

## Data Understanding
Dataset pada predictive analysis ini merupakan *Energy Consumption* yang memiliki 1000 data pada tahun 2022. [Energy Consumption Dataset](https://www.kaggle.com/datasets/mrsimple07/energy-consumption-prediction/code)
### Data Condition
- Tidak ada missing value pada data
- Tidak ada duplikasi pada data
- Terdapat Outliers pada Energy Consumption

### Variabel-variabel pada Energy Consumption Dataset:
| Fitur                       | Tipe        | Deskripsi                                             |
| --------------------------- | ----------- | ----------------------------------------------------- |
| TimeStamp                   | datetime    | waktu pengembilan data                                |
| Temperature                 | Numerical   | Temperatur                                            |
| Humidity                    | Numerical   | Kelembapan Udara                                      |
| SquareFootage               | Numerical   | Ukuran bangunan                                       |
| Occupancy                   | Numerical   | orang yang ada di dalam bangunan                                    |
| HVACUsage                 | Time        | Pengunaan Heating Ventilation AC                            |
|LightningUsage               |Categorical  | penggunaan lampu                                      |
|RenewableEnergy              |Numerical    | penggunaan energi terbarukan                          |
|DayOfWeek                    |Categorical  | hari dalam seminggu                                   |
|Holiday                      |Categorical  | representasi apakah hari itu libur atau tidak         |
|EnergyConsumption            |Numerical    | jumlah konsumsi energi                                |

## Exploratory Data Analysis
Visualisasi dan analisis untuk memahami distribusi dan hubungan antar fitur.
- **frekuensi setiap pada data kategorikal**
![alt text](https://jie25.s-ul.eu/aoBHYb7v)
![alt text](https://jie25.s-ul.eu/buMhx4G6)
![alt text](https://jie25.s-ul.eu/ax3aFRoy)
Dari data kategorikal yang disajikan terdapat insight bahwa penggunaan seperti lampu, HVAC, dan hari hari dimana terjadinya frekuensi tinggi permintaan energi seperti pada hari jumat, penggunaan lampu dan HVAC dengan mode on yang lebih sedikit


- **korelasi antara variable numerik**
![alt text](https://jie25.s-ul.eu/7YmcEmAH)
Dari korelasi antara variable numerik ini dapat disimpulkan bahwa temperatur dan occupancy menjadi beberapa faktor utama naiknya konsumsi energi

- **analisis hubungan antara DayOfWeek dan EnergyConsumption**
![alt text](https://jie25.s-ul.eu/UwwRajTs)
Dari data ini dapat disimpulkan bahwa hari jumat jumlah penggunaan energi paling banyak dari hari hari lain

## Data Preparation

### 1. Menyalin Dataset dengan dummy
- dengan adanya menyalin dataset ini, dataset asli aman dari perubahan berubahan
- menyelin dataset asli menjadi ```df_dummy```
- semua proses dan manipulasi ada pada ```df_dummy```

### 2. Penanganan Outliers
Outliers pada Energy Consumption di tangani dengan menggunakan IQR dimana angka pada data yang diambil ada di 25% sampai dengan 75% dengan itu outliers akan tidak ada dan data siap untuk di lakukan modelling

### 3. Penanganan Fitur yang tidak berguna
Berdasarkan EDA yang dilakukan timestamp, month dan year merupakan variable yang tidak terlalu berguna ketika proses modelling maka dari itu melakukan drop pada variable tersebut menjadi langkah yang bagus

### 4. Melakukan OneHotEncoding
- OneHotEncoding bertujuan untuk menjadikan data kategorikal menjadi numerikal/biner

### 5. Melakukan Standarisasi
- Melakukan Transformasi data numerik agar memiliki mean = 0 dan standart deviasi = 1

satu column yang tidak di dilakukan Standarisasi adalah ```EnergyConsumption``` karena akan menjadi fitur Y untuk splitting data

### 6. Splitting Data
- Fitur X: Gabungan dari kolom numerik dan kategorikal tetapi melakukan drop untuk column ```EnergyConsumption```
- Fitur Y: hanya mengambil ```EnergyConsumption```
- Train Test Split menggunakan rasio 80/20 menetapkan random_state untuk hasil yang reporducible

## Modeling
Pada Tahapan ini dilakukan prediksi pada energy consumption untuk memprediksi konsumsi energi menggunakan 5 algoritma regresson seperti SVR, GradientBoosting, XGboost, RandomForest, LightGBM

### 1. SVR (Support Vector Regression)
SVR merupakan cabang dari SVM yang dimana dapat menangkap model regresi untuk mengatasi masalah regresi

**Parameter**
paramater yang digunakan adalah parameter default untuk SVR

**Kelebihan**
- Dapat menangkap hubungan non linier yang baik dengan kernels
- Bagus untuk Dataset yang kecil

**Kekurangan**
- Tidak cocok dengan dataset besar
- Kurang bagus menintepretasi Fitur

### 2. RandomForest
Random Forest merupakan penggabungan beberapa decision tree atau ensemble dari decision tree yang mendapat prediksi akhir dari rata rata semua tree
**Parameter**
- ```random_state= 42``` 

**Kelebihan**
- menangkap hubungan non linear dan interaksi fitur
- Robust dari noise dan outliers

**Kelemahan**
- Algoritma tidak terlalu cepat

### 3. Gradient Boosting Regressor
Cara kerja gradient boosting yaitu dengan membuat tree secara sequential dimana setiap error di koreksi oleh tree baru

**Parameter**
-Default parameter dengan menggunakan ```random_state=42```

**Kelebihan**
- Lebih Akurat Dari RandomForest, tetapi lebih sensitif terhadap noise
- Bisa digunakan untuk model kompleks

**Kekurangan**
- Sensitif terhadap noise 
- Kurang Bagus untuk data yang besar

### 4. XGBoost
XGBoost merupakan versi yang lebih baik dari gradient boosting dengan menambahkan L1/L2 Regularisasi, Missing Value Handlingm dan Training secara parallel

**Parameter**
- Menggunakan Parameter default dengan ```random_state=42```

**Kelebihan**
- Fast and Scalable
- Menghandle missing value dengan baik

**Kekurangan**
- Memerlukan tuning hyperparameter yang kompleks.
- Memerlukan resource yang lebih banyak

### 5. LightGBM
Merupakan Boosting Algoritma yang membangun tree leaf-wise bukan level-wise

**Parameter**
- Menggunakan Parameter default dengan ```random_state=42```

**Kelebihan**
- Sangat cepat saat training di dataset besar
- sangat bagus untuk data kategorikal

**Kekurangan**
- Sangat mudah overfit
- Sensitive terhadap distribusi fitur

## Evaluation
Evaluasi digunakan untuk mencatat apakah ada kekurangan dan kelebihan pada model

### 1. Metriks Evaluasi
Metriks Evaluasi yang dilakukan adalah metrik untuk regresi
- MAE (Mean Absolute Error) : Menghitung rata rata absolut perbedaan antara prediksi dan aktual
- RMSE (Root Mean Squared Error) : mirip seperti MSE tetapi perbedaannya ada pada pengukuran RMSE mengukur unit yang sama pada target variable
- $R^2$ : Menghitung seberapa baik model menjelaskan varian pada variable target

### 2. Penjelasan Formula Metriks Evaluasi
- MAE : MAE menghitung nilai absolut dari prediksi dan aktual 
![alt text](https://arize.com/wp-content/uploads/2024/04/mean-absolute-error-formula.png)

    $y_i$ merupakan aktual sedangkan $\hat{y}_i$ merupakan prediksi dengan mengurangi dan membaginya dengan jumlah sample
- RMSE : Menghitung perbedaan nilai value yang di prediksi dan value yang di observasi/aktual
![alt text](https://arize.com/wp-content/uploads/2023/08/RMSE-equation.png)
dimana $y_i$ dikurangi $\hat{y}_i$ lalu dikuadratkan dan dibagi dengan jumlah sample lalu di akarkan untuk menghasilkan hasil perbedaan nilai antara prediksi dan value yang aktual

- $R^2$ : Menghitung seberapa baik model menjelaskan varian pada variable target
![alt text](https://vitalflux.com/wp-content/uploads/2019/07/R-squared-formula-function-of-SSE-and-SST.jpg)
$R^2$ menghitung 1 dikurang dengan total dari residual dibagi dengan total jumlah kuadrat

### 3. Hasil Perhitungan dari Metriks Evaluasi
| Model | MAE | RMSE | $R^2$|
|-------|-----|------|------|
|SVR    |4.148621|26.680706| 0.583187|
|RF     |4.046578	|27.273958|	0.573919|
|Gradient Boosting | 4.218489|	28.254645|	0.558598|
|XGBoost	|4.398486 |	33.320127|	0.479464|
|LightGBM	|4.282458	| 30.710678	|0.520229 |

**Relative MAE dan RMSE**
| MAE | RMSE |
|-----|------|
|5.46%|39.17%|

![alt image](https://jie25.s-ul.eu/w3TQddV7)
Disini dapat dilihat bahwa SVR memiliki overall performa terbaik dalam melakukan prediksi walaupun pada $R^2$ memiliki angka yang lebih tinggi dari dari keseluruhan algoritma SVR menunjukan angka yang lebih tinggi dari RF tetapi di RMSE SVR memiliki sedikit performa yang lebih baik dari RandomForest, dan pada $R^2$ memiliki angka yang lebih tinggi dari RandomForest dimana angka itu lebih mendekati 1 dari algoritma lain menunjukan SVR merupakan salah satu algoritma yang bagus untuk dataset ini, walaupun masih kurang bagus dalam memprediksi dimana angka RMSE yang tinggi perlu dilakukannya hyperparameter tuning


**Feature Importance**

Berdasarkan Hasil dari Feature Importance Random Forest dapat disimpulkan bahwa Temperature menjadi fitur yang menjadi faktor utama, selain itu ada renewable energy, Occupancy, Humidity, SquareFootage, dan Hour. Karena data ini juga berasal dari tahun 2022 dimana banyak orang yang masih bekerja dari rumah atau WFH maka tidak ada lonjakan seperti pada hari libur

![alt image](https://jie25.s-ul.eu/dHLvQOUd)

**Dampak ke Bussines Understanding**
- **Problem Statement and Goals**
- Problem statement 1:  Bagaimana Memprediksi penggunaan energi berdasarkan beberapa kondisi seperti temperatur, HVAC(Heating, Ventilation, and Air Conditioning), Karakteristik Bangunan, dan berapa banyak orang yang menempatinya? **Sudah Tercapai** walaupun terdapat kesalahan pada prediksi dan direkomendasi untuk menggunakan hyperparameter tuning untuk mencapai hasil yang diinginkan

- Problem Statement 2: Apa saja faktor terbesar dalam tingginya penggunaan energi? **Sudah Tercapai** dari analisis EDA dan Feature Importance dapat disimpulkan bahwa Temperatur menjadi faktor terbesar
