# filename: fontes_de_renda.py

import requests

def encontrar_criptomoeda(nome):
    url = f"https://api.coingecko.com/api/v3/coins/{nome}"
    resposta = requests.get(url)
    dados = resposta.json()
    return dados["market_data"]["current_price"]

criptomoeda = input("Qual é o nome da criptomoeda que você está procurando? ")
preco = encontrar_criptomoeda(criptomoeda)
print(f"O preço atual da {criptomoeda} é R${preco:.2f}")