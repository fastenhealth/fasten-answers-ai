import requests

def generar_comando_curl(query, k=3, threshold=0.2):
    query_encoded = requests.utils.quote(query)
    
    comando_curl = f'curl -N "http://localhost:8000/generate?query={query_encoded}&k={k}&threshold={threshold}"'
    return comando_curl

def main():
    while True:
        query = input("Enter your query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        k = 3
        threshold = 0.2
        
        comando_curl = generar_comando_curl(query, k, threshold)
        print("Generated curl command:")
        print(comando_curl)
        print()  

if __name__ == "__main__":
    main()
