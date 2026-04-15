# Calculus-App

Aplicación full-stack para práctica colaborativa de cálculo con un agente conversacional.

- **backend/** — API REST en Express + PostgreSQL + OpenAI
- **frontend/** — SPA en React (Vite)

## Requisitos

- Node.js ≥ 18.17
- Docker + Docker Compose
- Clave de API de OpenAI

## Puesta en marcha

```bash
# 1. Clonar y entrar al repo
git clone <url>
cd Calculus-App

# 2. Configurar variables de entorno
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
# edita backend/.env y pega tu OPENAI_API_KEY + un JWT_SECRET

# 3. Levantar PostgreSQL
cd backend
npm install
npm run db:up          # arranca postgres en localhost:5435 con el schema

# 4. Arrancar el backend (puerto 3002)
npm run dev

# 5. En otra terminal, arrancar el frontend (puerto 5173)
cd ../frontend
npm install
npm run dev
```

Abre http://localhost:5173.

## Scripts útiles

| Comando | Descripción |
|---|---|
| `npm run dev` (backend) | API con auto-reload |
| `npm run db:up` / `db:down` | Levanta/detiene Postgres en Docker |
| `npm run db:logs` | Logs del contenedor de DB |
| `npm run dev` (frontend) | Vite dev server |
| `npm run build` (frontend) | Build de producción |
| `npm run lint` (frontend) | ESLint |

## Arquitectura

```
backend/
  api/openapi.yml        # contrato OpenAPI (valida requests/responses)
  controllers/           # handlers HTTP
  data/                  # acceso a Postgres (pg)
  services/              # integración OpenAI (responde, analiza, filtra, etc.)
  middleware/            # auth JWT
  config/db.js           # pool de conexiones
  docker-compose.yml     # Postgres + init.sql
  data/init.sql          # schema inicial

frontend/
  src/
    components/          # componentes React
    pages/               # vistas
    services/            # cliente axios (usa VITE_API_URL)
    routes/              # rutas protegidas
```

## Autenticación

- `POST /login` y `POST /admin` devuelven `{ token, user }`
- El frontend guarda el token en localStorage y lo envía como `Authorization: Bearer <token>` en cada request
- Rutas protegidas en el backend usan `middleware/auth.js`

## Seed de admin

Al levantar la DB por primera vez, se inserta un admin:

- nickname: `admin`
- contraseña: `admin123` (hash bcrypt en `data/init.sql`)

Cámbialo antes de exponer cualquier instancia.

## Contribuir

1. Crea una rama desde `main`
2. `npm run lint` en `frontend/` debe pasar
3. Abre un PR con descripción clara del cambio
4. CI corre en GitHub Actions

## Licencia

TBD
