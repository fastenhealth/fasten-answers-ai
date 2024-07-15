import requests


def generate_curl_command(query, k=3, threshold=0.2):
    query_encoded = requests.utils.quote(query)
    
    curl_command = f'curl -N "http://localhost:8000/generate?query={query_encoded}&k={k}&threshold={threshold}"'
    return curl_command

def main():
    while True:
        query = input("Enter your query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        k = 3
        threshold = 0.2
        
        curl_command = generate_curl_command(query, k, threshold)
        print("Generated curl command:")
        print(curl_command)
        print()  

if __name__ == "__main__":
    main()
