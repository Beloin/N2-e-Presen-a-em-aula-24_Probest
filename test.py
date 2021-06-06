import pandas as pd

from regressao_linear import RegressaoLinearSimples
dados = pd.read_csv('./Dados semana 1 a 20 - Covid 2021 - Página1.csv')

impares = dados.loc[dados.Semana % 2 != 0]
X_impares = impares.Casos
Y_impares = impares.Obitos

model = RegressaoLinearSimples()
model.fit(X_impares.values ,Y_impares.values)

print(model.prever(450_000))