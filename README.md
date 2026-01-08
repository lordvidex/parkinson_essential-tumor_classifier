### 1. Generate YA_TOKEN
- Create a Yandex App to get client_id and client_secret: https://oauth.yandex.com/client/new
- Use the following URL to generate the token (replace client_id).
```
https://oauth.yandex.com/owl/authorize/error?response_type=token&client_id=954552f44e2a4e69ade4abf5ecc8bf2a&redirect_uri=https://oauth.yandex.ru/verification_code
```

### 2. Create .env file and add the following line:
YA_TOKEN=your_generated_token