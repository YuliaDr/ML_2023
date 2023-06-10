import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
import graphviz
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# page header
st.set_page_config(page_title='Анализ и прогнозирование временного ряда', page_icon=':books:', layout='wide')

# title
st.title('Анализ и прогнозирование временного ряда')
st.markdown(''' 
Это приложение позволяет произвести прогнозирование временного ряда для курса валют.

Загрузите xlsx файл с информацией о курсе с сайта ЦБ РФ и получите прогноз с помощью нескольких методов.

Ссылка на сайт: https://cbr.ru/currency_base/dynamics/
''')

# load data set
uploaded_file = st.file_uploader("Выберите файл")
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, header=0, index_col=1, parse_dates=True)

    df = df.drop(columns=['nominal', 'cdx'])

    fig1, ax = pyplot.subplots(1, 1, sharex='col', sharey='row', figsize=(10, 5))
    fig1.suptitle('Временной ряд в виде графика')
    df.plot(ax=ax, legend=False)
    st.pyplot(fig1)

    fig2, ax = pyplot.subplots(1, 1, sharex='col', sharey='row', figsize=(10, 5))
    fig2.suptitle('Гистограмма')
    df.hist(ax=ax, legend=False)
    st.pyplot(fig2)

    fig3, ax = pyplot.subplots(1, 1, sharex='col', sharey='row', figsize=(10, 5))
    fig3.suptitle('Плотность вероятности распределения данных')
    df.plot(ax=ax, kind='kde', legend=False)
    st.pyplot(fig3)

    df2 = df.copy()

    df2['SMA_10'] = df2['curs'].rolling(10, min_periods=1).mean()
    df2['SMA_20'] = df2['curs'].rolling(20, min_periods=1).mean()

    fig4, ax = pyplot.subplots(1, 1, sharex='col', sharey='row', figsize=(10, 5))
    fig4.suptitle('Временной ряд со скользящими средними')
    df2.plot(ax=ax, legend=True)
    st.pyplot(fig4)

    # разделение выборки на обучающую и тестовую
    xnum = list(range(df2.shape[0]))
    Y = df2['curs'].values
    train_size = int(len(Y) * 0.7)
    xnum_train, xnum_test = xnum[0:train_size], xnum[train_size:]
    train, test = Y[0:train_size], Y[train_size:]
    history_arima = [x for x in train]
    history_es = [x for x in train]

    st.sidebar.markdown('Выберите метод:')
    if st.sidebar.checkbox('ARIMA (интегрированная модель авторегрессии — скользящего среднего)'):

        st.header('ARIMA (интегрированная модель авторегрессии — скользящего среднего)')

        df3 = df2.copy()
        arima_order = (6, 1, 0)
        predictions_arima = list()

        for t in range(len(test)):
            model_arima = ARIMA(history_arima, order=arima_order)
            model_arima_fit = model_arima.fit()
            yhat_arima = model_arima_fit.forecast()[0]
            predictions_arima.append(yhat_arima)
            history_arima.append(test[t])

        df3['predictions_ARIMA'] = (train_size * [np.NAN]) + list(predictions_arima)

        fig5, ax = pyplot.subplots(1, 1, sharex='col', sharey='row', figsize=(20, 10))
        fig5.suptitle('Предсказания временного ряда')
        df3.plot(ax=ax, legend=True)
        st.pyplot(fig5)

        fig6, ax = pyplot.subplots(1, 1, sharex='col', sharey='row', figsize=(20, 10))
        fig6.suptitle('Предсказания временного ряда (тестовая выборка)')
        df3[train_size:].plot(ax=ax, legend=True)
        st.pyplot(fig6)

        st.markdown('Метод квадратичной ошибки:')
        st.markdown(mean_squared_error(test, predictions_arima, squared=False))
        st.markdown('Метод абсолютной ошибки:')
        st.markdown(mean_absolute_error(test, predictions_arima))

    if st.sidebar.checkbox('HWES (метода Хольта-Винтера)'):

        st.header('HWES (метода Хольта-Винтера)')

        df4 = df2.copy()

        # Формирование предсказаний
        predictions_es = list()
        for t in range(len(test)):
            model_es = ExponentialSmoothing(history_es)
            model_es_fit = model_es.fit()
            yhat_es = model_es_fit.forecast()[0]
            predictions_es.append(yhat_es)
            history_es.append(test[t])

        df4['predictions_HWES'] = (train_size * [np.NAN]) + list(predictions_es)

        fig7, ax = pyplot.subplots(1, 1, sharex='col', sharey='row', figsize=(20, 10))
        fig7.suptitle('Предсказания временного ряда')
        df4.plot(ax=ax, legend=True)
        st.pyplot(fig7)

        fig8, ax = pyplot.subplots(1, 1, sharex='col', sharey='row', figsize=(20, 10))
        fig8.suptitle('Предсказания временного ряда (тестовая выборка)')
        df4[train_size:].plot(ax=ax, legend=True)
        st.pyplot(fig8)

        st.markdown('Метод квадратичной ошибки:')
        st.markdown(mean_squared_error(test, predictions_es, squared=False))
        st.markdown('Метод абсолютной ошибки:')
        st.markdown(mean_absolute_error(test, predictions_es))