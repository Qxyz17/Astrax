# Astrax Web 聊天界面

一个本地运行的 React 聊天前端，后端是 llama.cpp 的 `llama-server`，
模型为 LFM2-350M。名字与图标均为 Astrax。

## 组成

| 组件 | 说明 |
|---|---|
| 前端 | React + Vite，`web/` |
| 后端 | `llama-server.exe`（OpenAI 兼容 API，`http://127.0.0.1:8080`） |
| 模型 | `data/LFM2-350M-Q4_K_M.gguf` |

前端通过 Vite 的 `/v1` 代理访问后端，浏览器无需处理 CORS。

## 一键启动

```bat
start_astrax.bat
```

然后在浏览器打开 **http://localhost:5173**。

## 手动启动

### 1. 后端

```bat
build\llama-cpu\bin\llama-server.exe -m data\LFM2-350M-Q4_K_M.gguf -t 4 --port 8080
```

### 2. 前端

```bat
cd web
npm install
npm run dev
```

## 生产构建

```bat
cd web
npm run build
```

产物在 `web/dist/`，可用任意静态服务器托管；
只要把 `/v1` 反向代理到 `http://127.0.0.1:8080` 即可。

## 自定义

- **名字**：`web/src/App.jsx` 里的 `Astrax`，以及 `web/index.html` 的 `<title>`。
- **图标**：`web/index.html` 的 `<link rel="icon">`（当前是 ◆ 符号，可换成图片）。
- **生成参数**：`web/src/App.jsx` 里 `streamChat` 的 `max_tokens`。
