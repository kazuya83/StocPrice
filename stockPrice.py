from sklearn.preprocessing import MinMaxScaler
from pandas_datareader import data
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from time import sleep
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pandas.tseries.offsets as offsets
import streamlit as st

class Model(nn.Module):
    def __init__(self, input=1, h=50, output=1):
        super().__init__()
        self.hidden_size = h
        
        self.lstm = nn.LSTM(input, h)
        self.fc = nn.Linear(h, output)
        self.hidden = (
            torch.zeros(1,1,h),
            torch.zeros(1,1,h)
        )
    
    def forward(self, seq):
        out, _ = self.lstm(
            seq.view(len(seq), 1, -1),
            self.hidden
        )
        out = self.fc(
            out.view(len(seq), -1)
        )
        return out[-1]

def ConvertTimestampToStringDate(timstamp):
    # Series datetime型に変換
    pdtimstamp = pd.Series(timstamp)
    # Series object型に変換
    seriesobject = pd.Series(pdtimstamp).astype(str)
    # 文字列に変換してreturn
    return seriesobject[0]

st.title('Stock Price Analysis')
brandCode = ""
text = st.text_input('Search for symbols or companies')
if text is not None and len(text) != 0:
    st.write('「' + text,'」 Searching ...')
    options = Options()
    options.add_argument('--headless')
    driver = webdriver.Chrome(chrome_options=options)
    driver.get('https://profile.yahoo.co.jp/')
    search = driver.find_element_by_id('searchTextCom')
    search.send_keys(text)
    driver.find_element_by_id('searchButtonCom').click()
    sleep(1)
    count = len(driver.find_elements_by_class_name('yjL'))
    if count > 0:
        isGet = True
        brandInfoStr = "Company Infomation：" + driver.find_element_by_class_name('yjL').text
    else:
        isGet = False
        brandInfoStr = "Not Found"
    st.write(brandInfoStr)
    driver.quit()
    if isGet:
        strLen = len(brandInfoStr)
        brandCode = brandInfoStr[(strLen - 5):(strLen - 1)]
        st.write("Symbol code：" + brandCode)
        
if len(brandCode) != 0:
    stock_data = data.DataReader(brandCode + '.JP', 'stooq').sort_values('Date', ascending=True)
    st.write('Current stock price')
    stock_data
    stock_data = stock_data.drop(['Open', 'High', 'Low', 'Volume'], axis=1)
    st.line_chart(stock_data)
    # データの正規化
    y = stock_data['Close'].values
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler.fit(y.reshape(-1,1))
    y = scaler.transform(y.reshape(-1,1))
    y = torch.FloatTensor(y).view(-1)
    
    train_window_size = 7
    def input_data(seq, ws):
        out = []
        L = len(seq)
        for i in range(L-ws):
            window = seq[i:i+ws]
            label = seq[i+ws:i+ws+1]
            out.append((window, label))
        return out

    # 直近までの全てのデータをトレーニング用としてモデルに渡す
    train_data = input_data(y, train_window_size)
    
    torch.manual_seed(123)
    model = Model()
    
    # mean auqrer error loss 平均二乗誤作法
    criterion = nn.MSELoss()
    
    # stocastic gradient descent 確率的勾配降下法
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    def run_train():
        for train_window, correct_label in train_data:
            optimizer.zero_grad()
            
            model.hidden = (
                torch.zeros(1,1,model.hidden_size),
                torch.zeros(1,1,model.hidden_size)
            )
            
            train_predicted_label = model.forward(train_window)
            train_loss = criterion(train_predicted_label,correct_label)
            
            train_loss.backward()
            optimizer.step()
            
    test_size = 30
    
    def run_test():
        for i in range(test_size):
            test_window = torch.FloatTensor(extending_seq[-test_size:])
            
            with torch.no_grad():
                model.hidden = (
                    torch.zeros(1,1,model.hidden_size),
                    torch.zeros(1,1,model.hidden_size)
                )
                test_predicted_label = model.forward(test_window)
                extending_seq.append(test_predicted_label.item())
                
    epochs = 20
    
    st.write('Machine learing...')
    for epoch in range(epochs):
        print()
        print(f'Epoch: {epoch+1}')
        
        run_train()
        
        extending_seq = y[-test_size:].tolist()
        
        run_test()
    
    # 未来のデータの数値を、株価のスケールに変換する
    predicted_normalized_labels_list = extending_seq[-test_size:]
    predicted_normalized_labels_array_1d = np.array(predicted_normalized_labels_list)
    predicted_normalized_labels_array_2d = predicted_normalized_labels_array_1d.reshape(-1,1)
    predicted_labels_array_2d = scaler.inverse_transform(predicted_normalized_labels_array_2d)
    
    # 直近データの最終日
    real_last_date_timestamp = stock_data.index[-1]
    # 未来日の最初の日付(str型)
    future_first_date_str = ConvertTimestampToStringDate(real_last_date_timestamp + offsets.Day())
    # 未来の最後の翌日の日付
    second_argument_date_str = ConvertTimestampToStringDate(real_last_date_timestamp + offsets.Day(31))
    future_period = np.arange(future_first_date_str, second_argument_date_str, dtype='datetime64')
    
    # 未来予測のデータをpandasデータに変換
    predict_pd = pd.DataFrame(
        {'Close':np.ravel(predicted_labels_array_2d)}
        ,index=np.ravel(future_period)
    )
    st.write()
    st.write('Stock price forecast')
    predict_pd
    for idx in range(len(future_period)):
        stock_data.loc[future_period[idx]] = predicted_labels_array_2d[idx]
    # stock_data = stock_data.append(predict_pd)
    
    st.line_chart(stock_data)
    
    

