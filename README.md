## Steps to Run the Application

1. Install **ollama** and then run:
   ```bash
   ollama run llama3.1

2. Activate the virtual environment:

    ```bash
    source venv/bin/activate

3. Run the application:
    ```bash
    python3 app.py

4. Open Postman or use curl to send the following request:
    ```bash
    curl --location 'http://localhost:8080/ask_pdf' \
    --header 'Content-Type: application/json' \
    --data '{
        "query":"Who is Alice?"
    }'


License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)