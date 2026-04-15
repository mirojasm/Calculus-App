# Calculus-App

Aplicación full-stack para práctica colaborativa de cálculo con un agente conversacional.

- **backend/** — API REST (Express + PostgreSQL + OpenAI)
- **frontend/** — SPA en React (Vite)

## Arranque rápido (3 comandos)

```bash
git clone <url> && cd Calculus-App

npm install        # instala backend y frontend (npm workspaces)
npm run setup      # crea los .env y te pide tu OPENAI_API_KEY
npm start          # levanta Postgres (Docker) + backend + frontend
```

Abre **http://localhost:5173** y listo.

> Requisitos: Node ≥ 18.17 + Docker Desktop corriendo.

## Credenciales por defecto

- **Admin** — nickname: `admin`, contraseña: `admin123` (cámbialo antes de exponer la app)
- **Usuario** — regístrate desde la UI

## Scripts útiles (desde la raíz)

| Comando | Descripción |
|---|---|
| `npm run dev` | Backend + frontend en paralelo |
| `npm run dev:backend` | Solo backend (puerto 3002) |
| `npm run dev:frontend` | Solo frontend (puerto 5173) |
| `npm run db:up` / `db:down` | Levanta/detiene Postgres (puerto 5435) |
| `npm run db:logs` | Tail de logs de Postgres |
| `npm run build` | Build de producción del frontend |
| `npm run lint` | ESLint sobre el frontend |
| `npm start` | `db:up` + `dev` |

## Arquitectura

```
backend/
  api/openapi.yml        contrato OpenAPI
  config/db.js           pool pg
  controllers/           handlers HTTP
  data/                  acceso a Postgres
  data/init.sql          schema inicial
  middleware/auth.js     JWT (authRequired, adminRequired)
  services/              integración OpenAI (responde, analiza, filtra...)
  docker-compose.yml     Postgres (+ backend con profile=full)
  Dockerfile             imagen del backend

frontend/
  src/
    services/            cliente axios con VITE_API_URL + interceptor Bearer
    components/ pages/   UI
```

## Autenticación

- `POST /login` y `POST /admin` devuelven `{ token, user }`
- El frontend guarda el token en `localStorage` y el `apiClient` lo envía como `Authorization: Bearer <token>` en cada request
- Rutas públicas: `POST /login`, `POST /admin`, `POST /usuario` (registro), `GET /health`
- Todo lo demás exige token válido (middleware `authRequired`)

## Variables de entorno

Plantillas en `backend/.env.example` y `frontend/.env.example`. El script `npm run setup` las copia automáticamente y genera un `JWT_SECRET` aleatorio.

Lo único imprescindible de editar es `OPENAI_API_KEY` en `backend/.env`.

## Correr todo en Docker (sin Node en el host)

```bash
cp backend/.env.example backend/.env  # edita OPENAI_API_KEY
cd backend
docker compose --profile full up --build
```

## Contribuir

1. Crea una rama desde `main`
2. `npm run lint` debe pasar
3. CI corre automáticamente en GitHub Actions
4. Abre un PR con descripción clara
